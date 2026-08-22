"""Sample file discovery for Lora and Sampling output directories."""

from pathlib import Path


def resolve_safe_sample_file(base: Path, relative_path: str) -> Path | None:
    """Resolve a sample path under base, or None if missing / outside base."""
    resolved_base = base.resolve()
    target = (resolved_base / relative_path).resolve()
    if not target.is_relative_to(resolved_base) or not target.is_file():
        return None
    return target


def list_samples_for_output_dir(
    output_dir: Path,
    *,
    sampling_id: int | None = None,
) -> list[tuple[Path, str, dict]]:
    """Return (path, kind, metadata) tuples for sample files."""
    if not output_dir.exists():
        return []

    if sampling_id is not None:
        grid_prefix = f"{sampling_id}_grid_"
        flat_samples = sorted(
            path
            for path in output_dir.glob(f"{sampling_id}_*.png")
            if not path.name.startswith(grid_prefix)
        )
        flat_grids = sorted(output_dir.glob(f"{grid_prefix}*.png"))
        if flat_samples or flat_grids:
            results: list[tuple[Path, str, dict]] = [(path, "cell", {}) for path in flat_samples]
            results.extend((path, "grid", {}) for path in flat_grids)
            return results

    from src.sampler.sweep.manifest import GRIDS_SUBDIR, IMAGES_SUBDIR, read_manifest

    manifest = read_manifest(output_dir)
    if manifest is not None:
        results: list[tuple[Path, str, dict]] = []
        for grid in manifest.grids:
            path = output_dir / grid.file
            if path.is_file():
                results.append((path, "grid", {"title": grid.title, "index": grid.index}))
        for image in manifest.images:
            path = output_dir / image.file
            if path.is_file():
                results.append((path, "cell", {"params": image.params, "index": image.index}))
        return results

    results: list[tuple[Path, str, dict]] = []
    images_dir = output_dir / IMAGES_SUBDIR
    if images_dir.is_dir():
        results.extend((path, "cell", {}) for path in sorted(images_dir.glob("*.png")))
    grids_dir = output_dir / GRIDS_SUBDIR
    if grids_dir.is_dir():
        results.extend((path, "grid", {}) for path in sorted(grids_dir.glob("*.png")))
    if results:
        return results

    samples_dir = output_dir / "samples"
    if samples_dir.is_dir():
        return [(path, "legacy", {}) for path in sorted(samples_dir.glob("*.png"))]
    return [(path, "legacy", {}) for path in sorted(output_dir.glob("*.png"))]
