"""Sweep manifest read/write."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ManifestImageEntry(BaseModel):
    index: int
    file: str
    params: dict[str, Any] = Field(default_factory=dict)
    grid_position: dict[str, int] | None = None


class ManifestGridAxis(BaseModel):
    param: str
    values: list[Any] = Field(default_factory=list)


class ManifestGridEntry(BaseModel):
    index: int
    file: str
    slice: dict[str, Any] = Field(default_factory=dict)
    x: ManifestGridAxis
    y: ManifestGridAxis
    cells: list[list[int | None]] = Field(default_factory=list)
    title: str = ""


class SweepManifest(BaseModel):
    version: int = 1
    config_id: int | None = None
    job_id: int | None = None
    total_images: int = 0
    images: list[ManifestImageEntry] = Field(default_factory=list)
    grids: list[ManifestGridEntry] = Field(default_factory=list)


SampleKind = Literal["cell", "grid", "legacy"]


MANIFEST_FILENAME = "manifest.json"
IMAGES_SUBDIR = "images"
GRIDS_SUBDIR = "grids"


def manifest_path(output_dir: Path) -> Path:
    return output_dir / MANIFEST_FILENAME


def write_manifest(output_dir: Path, manifest: SweepManifest) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(output_dir)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def read_manifest(output_dir: Path) -> SweepManifest | None:
    path = manifest_path(output_dir)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return SweepManifest.model_validate(data)


def cell_image_path(output_dir: Path, index: int) -> Path:
    return output_dir / IMAGES_SUBDIR / f"cell_{index:04d}.png"


def grid_image_path(output_dir: Path, index: int, title: str = "") -> Path:
    suffix = ""
    if title:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in title)[:60]
        suffix = f"_{safe}"
    return output_dir / GRIDS_SUBDIR / f"grid_{index:03d}{suffix}.png"
