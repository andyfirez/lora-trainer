"""Sample file discovery for job and trained LoRA output directories."""

from pathlib import Path


def list_samples_for_output_dir(output_dir: Path) -> list[tuple[Path, str, dict]]:
    """Return (path, kind, metadata) tuples for sample files."""
    if not output_dir.exists():
        return []

    from src.sampler.sweep.manifest import read_manifest

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

    samples_dir = output_dir / "samples"
    if samples_dir.is_dir():
        return [(path, "legacy", {}) for path in sorted(samples_dir.glob("*.png"))]
    return [(path, "legacy", {}) for path in sorted(output_dir.glob("*.png"))]
