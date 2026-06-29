"""Dataset image preprocessing: fit, crop, bake, and validation."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from PIL import Image

from src.db.tables.dataset import Dataset
from src.services.datasets.captions import list_image_filenames

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_MIN_RESOLUTION = 64
_MAX_RESOLUTION = 2048


class ImagePreprocessState(StrEnum):
    NO_CROP = "no_crop"
    STALE = "stale"
    CROPPED = "cropped"
    READY = "ready"


@dataclass(frozen=True)
class FittedSize:
    width: int
    height: int


@dataclass(frozen=True)
class PreprocessStatus:
    target_resolution: int | None
    preprocess_ready: bool
    total: int
    no_crop: int
    stale: int
    cropped: int
    ready: int


@dataclass(frozen=True)
class CropMeta:
    crop_center_x: float
    crop_center_y: float
    fitted_width: int
    fitted_height: int
    source_width: int
    source_height: int
    state: ImagePreprocessState


def prepared_dir_path(image_dir: str | Path, resolution: int) -> Path:
    return Path(image_dir) / ".prepared" / str(resolution)


def validate_target_resolution(resolution: int) -> None:
    if resolution < _MIN_RESOLUTION or resolution > _MAX_RESOLUTION:
        raise ValueError(f"target_resolution must be between {_MIN_RESOLUTION} and {_MAX_RESOLUTION}")


def _fit_size(width: int, height: int, resolution: int) -> FittedSize:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    scale = resolution / min(width, height)
    return FittedSize(
        width=max(int(round(width * scale)), 1),
        height=max(int(round(height * scale)), 1),
    )


def _clamp_crop_center(
    center_x: float,
    center_y: float,
    fitted: FittedSize,
    resolution: int,
) -> tuple[float, float]:
    half = resolution / 2.0
    min_cx = half / fitted.width
    max_cx = 1.0 - half / fitted.width if fitted.width > resolution else 0.5
    min_cy = half / fitted.height
    max_cy = 1.0 - half / fitted.height if fitted.height > resolution else 0.5
    if fitted.width <= resolution:
        cx = 0.5
    else:
        cx = min(max(center_x, min_cx), max_cx)
    if fitted.height <= resolution:
        cy = 0.5
    else:
        cy = min(max(center_y, min_cy), max_cy)
    return cx, cy


def _crop_box(
    fitted: FittedSize,
    resolution: int,
    center_x: float,
    center_y: float,
) -> tuple[int, int, int, int]:
    cx, cy = _clamp_crop_center(center_x, center_y, fitted, resolution)
    center_px_x = cx * fitted.width
    center_px_y = cy * fitted.height
    left = int(round(center_px_x - resolution / 2))
    top = int(round(center_px_y - resolution / 2))
    left = max(0, min(left, fitted.width - resolution))
    top = max(0, min(top, fitted.height - resolution))
    return left, top, left + resolution, top + resolution


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _resize_to_fit(image: Image.Image, resolution: int) -> Image.Image:
    fitted = _fit_size(image.width, image.height, resolution)
    if image.width == fitted.width and image.height == fitted.height:
        return image.copy()
    return image.resize((fitted.width, fitted.height), Image.Resampling.LANCZOS)


def apply_crop(
    image: Image.Image,
    resolution: int,
    center_x: float,
    center_y: float,
) -> Image.Image:
    fitted = _fit_size(image.width, image.height, resolution)
    resized = _resize_to_fit(image, resolution)
    box = _crop_box(fitted, resolution, center_x, center_y)
    cropped = resized.crop(box)
    if cropped.size != (resolution, resolution):
        raise ValueError(f"Cropped image size {cropped.size} != ({resolution}, {resolution})")
    return cropped


def default_crop_center(width: int, height: int) -> tuple[float, float]:
    return 0.5, 0.5


def source_mtime(path: Path) -> float:
    return path.stat().st_mtime


def is_crop_stale(crop_mtime: float, current_mtime: float) -> bool:
    return current_mtime > crop_mtime + 1e-6


def prepared_image_path(prepared_dir: Path, filename: str) -> Path:
    return prepared_dir / filename


def resolve_prepared_path(prepared_dir: Path, filename: str) -> Path | None:
    prepared_path = prepared_dir / filename
    if prepared_path.is_file():
        return prepared_path
    alt_png = prepared_dir / f"{Path(filename).stem}.png"
    if alt_png.is_file():
        return alt_png
    return None


def is_prepared_file_valid(prepared_path: Path, resolution: int) -> bool:
    if not prepared_path.is_file():
        return False
    try:
        with Image.open(prepared_path) as img:
            return img.size == (resolution, resolution)
    except OSError:
        return False


def get_image_state(
    *,
    filename: str,
    image_dir: Path,
    resolution: int | None,
    crop_mtime: float | None,
    crop_baked_at: datetime | None,
) -> ImagePreprocessState:
    if resolution is None:
        return ImagePreprocessState.NO_CROP
    path = image_dir / filename
    if not path.is_file():
        return ImagePreprocessState.NO_CROP
    if crop_mtime is None:
        return ImagePreprocessState.NO_CROP
    current_mtime = source_mtime(path)
    if is_crop_stale(crop_mtime, current_mtime):
        return ImagePreprocessState.STALE
    prepared_path = prepared_image_path(prepared_dir_path(image_dir, resolution), filename)
    if crop_baked_at is None or not is_prepared_file_valid(prepared_path, resolution):
        return ImagePreprocessState.CROPPED
    return ImagePreprocessState.READY


def build_crop_meta(
    *,
    image_path: Path,
    resolution: int,
    crop_center_x: float | None,
    crop_center_y: float | None,
    crop_mtime: float | None,
    crop_baked_at: datetime | None,
) -> CropMeta:
    with Image.open(image_path) as img:
        source_width, source_height = img.size
    fitted = _fit_size(source_width, source_height, resolution)
    if crop_center_x is None or crop_center_y is None:
        cx, cy = default_crop_center(source_width, source_height)
        state = ImagePreprocessState.NO_CROP
    else:
        cx, cy = _clamp_crop_center(crop_center_x, crop_center_y, fitted, resolution)
        state = get_image_state(
            filename=image_path.name,
            image_dir=image_path.parent,
            resolution=resolution,
            crop_mtime=crop_mtime,
            crop_baked_at=crop_baked_at,
        )
    return CropMeta(
        crop_center_x=cx,
        crop_center_y=cy,
        fitted_width=fitted.width,
        fitted_height=fitted.height,
        source_width=source_width,
        source_height=source_height,
        state=state,
    )


def bake_image_to_prepared(
    *,
    source_path: Path,
    prepared_dir: Path,
    resolution: int,
    center_x: float,
    center_y: float,
) -> Path:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    image = _load_rgb(source_path)
    result = apply_crop(image, resolution, center_x, center_y)
    output_path = prepared_image_path(prepared_dir, source_path.name)
    suffix = source_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        result.save(output_path, format="JPEG", quality=95)
    elif suffix == ".png":
        result.save(output_path, format="PNG")
    elif suffix == ".webp":
        result.save(output_path, format="WEBP", quality=95)
    else:
        result.save(output_path, format="PNG")
    return output_path


def build_fitted_preview_bytes(source_path: Path, resolution: int) -> bytes:
    from io import BytesIO

    image = _load_rgb(source_path)
    fitted = _resize_to_fit(image, resolution)
    buffer = BytesIO()
    fitted.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def compute_preprocess_status(
    dataset: Dataset,
    crop_by_filename: dict[str, tuple[float, float, float, datetime | None]],
) -> PreprocessStatus:
    image_dir = Path(dataset.image_dir)
    filenames = list_image_filenames(image_dir)
    resolution = dataset.target_resolution
    counts = {state: 0 for state in ImagePreprocessState}
    for filename in filenames:
        crop_data = crop_by_filename.get(filename)
        if crop_data is None:
            state = ImagePreprocessState.NO_CROP if resolution else ImagePreprocessState.NO_CROP
            if resolution is None:
                state = ImagePreprocessState.NO_CROP
            else:
                state = ImagePreprocessState.NO_CROP
        else:
            center_x, center_y, mtime, baked_at = crop_data
            state = get_image_state(
                filename=filename,
                image_dir=image_dir,
                resolution=resolution,
                crop_mtime=mtime,
                crop_baked_at=baked_at,
            )
        counts[state] += 1
    return PreprocessStatus(
        target_resolution=resolution,
        preprocess_ready=dataset.preprocess_ready,
        total=len(filenames),
        no_crop=counts[ImagePreprocessState.NO_CROP],
        stale=counts[ImagePreprocessState.STALE],
        cropped=counts[ImagePreprocessState.CROPPED],
        ready=counts[ImagePreprocessState.READY],
    )


def recompute_preprocess_ready(
    dataset: Dataset,
    crop_by_filename: dict[str, tuple[float, float, float, datetime | None]],
) -> bool:
    status = compute_preprocess_status(dataset, crop_by_filename)
    if status.total == 0 or dataset.target_resolution is None:
        return False
    return status.ready == status.total
