"""Tests for sweep manifest IO."""

from pathlib import Path

from src.sampler.sweep.manifest import (
    ManifestGridAxis,
    ManifestGridEntry,
    ManifestImageEntry,
    SweepManifest,
    read_manifest,
    write_manifest,
)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = SweepManifest(
        job_id=1,
        total_images=2,
        images=[
            ManifestImageEntry(index=0, file="images/cell_0000.png", params={"prompt": "a"}),
            ManifestImageEntry(index=1, file="images/cell_0001.png", params={"prompt": "b"}),
        ],
        grids=[
            ManifestGridEntry(
                index=0,
                file="grids/grid_000.png",
                x=ManifestGridAxis(param="prompt", values=["a", "b"]),
                y=ManifestGridAxis(param="lora_weight", values=[1.0]),
                cells=[[0, 1]],
            )
        ],
    )
    write_manifest(tmp_path, manifest)
    loaded = read_manifest(tmp_path)
    assert loaded is not None
    assert loaded.total_images == 2
    assert len(loaded.grids) == 1
