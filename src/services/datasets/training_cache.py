"""Training disk cache invalidation for dataset images."""

from pathlib import Path

from src.services.datasets.preprocess import prepared_dir_path, resolve_prepared_path

_TE_NPZ_SUFFIX = "_te.npz"


def invalidate_te_cache_for_image(
    image_dir: str | Path,
    filename: str,
    target_resolution: int | None,
) -> None:
    if target_resolution is None:
        return
    prepared_dir = prepared_dir_path(image_dir, target_resolution)
    prepared_path = resolve_prepared_path(prepared_dir, filename)
    if prepared_path is None:
        return
    cache_path = prepared_dir / f"{prepared_path.stem}{_TE_NPZ_SUFFIX}"
    try:
        cache_path.unlink(missing_ok=True)
    except OSError:
        pass
