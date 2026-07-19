"""Dataset image preprocessing: fit, crop, bake, and validation."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from PIL import Image

from src.db.tables.dataset import Dataset
from src.services.datasets.captions import list_image_filenames
from src.services.datasets.formats import IMAGE_EXTENSIONS, PREPARED_EXTENSION
from src.trainer.sdxl.buckets import (
    BucketAssignment,
    assign_bucket,
    assignment_from_stored,
)

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
class BucketPreprocessConfig:
    enable_bucket: bool
    resolution: int
    min_bucket_reso: int
    max_bucket_reso: int
    bucket_reso_steps: int
    bucket_no_upscale: bool

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> "BucketPreprocessConfig | None":
        if dataset.target_resolution is None:
            return None
        return cls(
            enable_bucket=dataset.enable_bucket,
            resolution=dataset.target_resolution,
            min_bucket_reso=dataset.min_bucket_reso,
            max_bucket_reso=dataset.max_bucket_reso,
            bucket_reso_steps=dataset.bucket_reso_steps,
            bucket_no_upscale=dataset.bucket_no_upscale,
        )


@dataclass(frozen=True)
class StoredCropRecord:
    crop_center_x: float
    crop_center_y: float
    source_mtime: float
    baked_at: datetime | None
    bucket_width: int | None = None
    bucket_height: int | None = None
    scale_to_width: int | None = None
    scale_to_height: int | None = None
    crop_x: int = 0
    crop_y: int = 0


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
    enable_bucket: bool = False
    bucket_width: int | None = None
    bucket_height: int | None = None
    scale_to_width: int | None = None
    scale_to_height: int | None = None
    crop_x: int = 0
    crop_y: int = 0


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


def _resolve_assignment(
    *,
    source_width: int,
    source_height: int,
    bucket_config: BucketPreprocessConfig,
    center_x: float,
    center_y: float,
    stored: StoredCropRecord | None,
) -> BucketAssignment:
    if (
        stored is not None
        and stored.bucket_width is not None
        and stored.bucket_height is not None
        and stored.scale_to_width is not None
        and stored.scale_to_height is not None
    ):
        return assignment_from_stored(
            source_width=source_width,
            source_height=source_height,
            bucket_width=stored.bucket_width,
            bucket_height=stored.bucket_height,
            scale_to_width=stored.scale_to_width,
            scale_to_height=stored.scale_to_height,
            crop_x=stored.crop_x,
            crop_y=stored.crop_y,
        )
    return assign_bucket(
        source_width,
        source_height,
        resolution=bucket_config.resolution,
        min_bucket_reso=bucket_config.min_bucket_reso,
        max_bucket_reso=bucket_config.max_bucket_reso,
        bucket_reso_steps=bucket_config.bucket_reso_steps,
        bucket_no_upscale=bucket_config.bucket_no_upscale,
        center_x=center_x,
        center_y=center_y,
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


def _clamp_bucket_crop_center(
    center_x: float,
    center_y: float,
    assignment: BucketAssignment,
) -> tuple[float, float]:
    scale_w = assignment.scale_to_width
    scale_h = assignment.scale_to_height
    bucket_w = assignment.bucket_width
    bucket_h = assignment.bucket_height
    if scale_w <= bucket_w:
        cx = 0.5
    else:
        half = bucket_w / 2.0
        min_cx = half / scale_w
        max_cx = 1.0 - half / scale_w
        cx = min(max(center_x, min_cx), max_cx)
    if scale_h <= bucket_h:
        cy = 0.5
    else:
        half = bucket_h / 2.0
        min_cy = half / scale_h
        max_cy = 1.0 - half / scale_h
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


def apply_bucket_crop(
    image: Image.Image,
    assignment: BucketAssignment,
) -> Image.Image:
    resized = image.resize(
        (assignment.scale_to_width, assignment.scale_to_height),
        Image.Resampling.LANCZOS,
    )
    box = (
        assignment.crop_x,
        assignment.crop_y,
        assignment.crop_x + assignment.bucket_width,
        assignment.crop_y + assignment.bucket_height,
    )
    cropped = resized.crop(box)
    expected = (assignment.bucket_width, assignment.bucket_height)
    if cropped.size != expected:
        raise ValueError(f"Cropped image size {cropped.size} != {expected}")
    return cropped


def default_crop_center(width: int, height: int) -> tuple[float, float]:
    return 0.5, 0.5


def source_mtime(path: Path) -> float:
    return path.stat().st_mtime


def is_crop_stale(crop_mtime: float, current_mtime: float) -> bool:
    return current_mtime > crop_mtime + 1e-6


def prepared_jpg_path(prepared_dir: Path, source_filename: str) -> Path:
    return prepared_dir / f"{Path(source_filename).stem}{PREPARED_EXTENSION}"


def prepared_image_path(prepared_dir: Path, filename: str) -> Path:
    return prepared_jpg_path(prepared_dir, filename)


def resolve_prepared_path(prepared_dir: Path, filename: str) -> Path | None:
    prepared_path = prepared_dir / filename
    if prepared_path.is_file():
        return prepared_path
    jpg_path = prepared_jpg_path(prepared_dir, filename)
    if jpg_path.is_file():
        return jpg_path
    alt_png = prepared_dir / f"{Path(filename).stem}.png"
    if alt_png.is_file():
        return alt_png
    return None


def _cleanup_stale_prepared(prepared_dir: Path, stem: str, keep_path: Path) -> None:
    for ext in IMAGE_EXTENSIONS:
        candidate = prepared_dir / f"{stem}{ext}"
        if candidate == keep_path or not candidate.is_file():
            continue
        invalidate_latent_cache_for_prepared(candidate)
        candidate.unlink(missing_ok=True)


def is_prepared_file_valid(
    prepared_path: Path,
    expected_width: int,
    expected_height: int,
) -> bool:
    if not prepared_path.is_file():
        return False
    try:
        with Image.open(prepared_path) as img:
            return img.size == (expected_width, expected_height)
    except OSError:
        return False


def expected_prepared_size(
    *,
    bucket_config: BucketPreprocessConfig,
    source_width: int,
    source_height: int,
    center_x: float,
    center_y: float,
    stored: StoredCropRecord | None,
) -> tuple[int, int]:
    if bucket_config.enable_bucket:
        assignment = _resolve_assignment(
            source_width=source_width,
            source_height=source_height,
            bucket_config=bucket_config,
            center_x=center_x,
            center_y=center_y,
            stored=stored,
        )
        return assignment.bucket_width, assignment.bucket_height
    resolution = bucket_config.resolution
    return resolution, resolution


def get_image_state(
    *,
    filename: str,
    image_dir: Path,
    bucket_config: BucketPreprocessConfig | None,
    crop_record: StoredCropRecord | None,
) -> ImagePreprocessState:
    if bucket_config is None:
        return ImagePreprocessState.NO_CROP
    path = image_dir / filename
    if not path.is_file():
        return ImagePreprocessState.NO_CROP
    if crop_record is None:
        return ImagePreprocessState.NO_CROP
    current_mtime = source_mtime(path)
    if is_crop_stale(crop_record.source_mtime, current_mtime):
        return ImagePreprocessState.STALE
    with Image.open(path) as img:
        source_width, source_height = img.size
    expected_w, expected_h = expected_prepared_size(
        bucket_config=bucket_config,
        source_width=source_width,
        source_height=source_height,
        center_x=crop_record.crop_center_x,
        center_y=crop_record.crop_center_y,
        stored=crop_record,
    )
    prepared_path = resolve_prepared_path(
        prepared_dir_path(image_dir, bucket_config.resolution),
        filename,
    )
    if (
        prepared_path is None
        or crop_record.baked_at is None
        or not is_prepared_file_valid(prepared_path, expected_w, expected_h)
    ):
        return ImagePreprocessState.CROPPED
    return ImagePreprocessState.READY


def build_crop_meta(
    *,
    image_path: Path,
    bucket_config: BucketPreprocessConfig,
    crop_center_x: float | None,
    crop_center_y: float | None,
    stored: StoredCropRecord | None,
) -> CropMeta:
    with Image.open(image_path) as img:
        source_width, source_height = img.size

    if bucket_config.enable_bucket:
        if crop_center_x is None or crop_center_y is None:
            cx, cy = default_crop_center(source_width, source_height)
            state = ImagePreprocessState.NO_CROP
            assignment = assign_bucket(
                source_width,
                source_height,
                resolution=bucket_config.resolution,
                min_bucket_reso=bucket_config.min_bucket_reso,
                max_bucket_reso=bucket_config.max_bucket_reso,
                bucket_reso_steps=bucket_config.bucket_reso_steps,
                bucket_no_upscale=bucket_config.bucket_no_upscale,
                center_x=cx,
                center_y=cy,
            )
        else:
            assignment = _resolve_assignment(
                source_width=source_width,
                source_height=source_height,
                bucket_config=bucket_config,
                center_x=crop_center_x,
                center_y=crop_center_y,
                stored=stored,
            )
            cx, cy = _clamp_bucket_crop_center(crop_center_x, crop_center_y, assignment)
            crop_record = StoredCropRecord(
                crop_center_x=cx,
                crop_center_y=cy,
                source_mtime=stored.source_mtime if stored else source_mtime(image_path),
                baked_at=stored.baked_at if stored else None,
                bucket_width=assignment.bucket_width,
                bucket_height=assignment.bucket_height,
                scale_to_width=assignment.scale_to_width,
                scale_to_height=assignment.scale_to_height,
                crop_x=assignment.crop_x,
                crop_y=assignment.crop_y,
            )
            state = get_image_state(
                filename=image_path.name,
                image_dir=image_path.parent,
                bucket_config=bucket_config,
                crop_record=crop_record,
            )
        return CropMeta(
            crop_center_x=cx,
            crop_center_y=cy,
            fitted_width=assignment.scale_to_width,
            fitted_height=assignment.scale_to_height,
            source_width=source_width,
            source_height=source_height,
            state=state,
            enable_bucket=True,
            bucket_width=assignment.bucket_width,
            bucket_height=assignment.bucket_height,
            scale_to_width=assignment.scale_to_width,
            scale_to_height=assignment.scale_to_height,
            crop_x=assignment.crop_x,
            crop_y=assignment.crop_y,
        )

    resolution = bucket_config.resolution
    fitted = _fit_size(source_width, source_height, resolution)
    if crop_center_x is None or crop_center_y is None:
        cx, cy = default_crop_center(source_width, source_height)
        state = ImagePreprocessState.NO_CROP
    else:
        cx, cy = _clamp_crop_center(crop_center_x, crop_center_y, fitted, resolution)
        crop_record = StoredCropRecord(
            crop_center_x=cx,
            crop_center_y=cy,
            source_mtime=stored.source_mtime if stored else source_mtime(image_path),
            baked_at=stored.baked_at if stored else None,
        )
        state = get_image_state(
            filename=image_path.name,
            image_dir=image_path.parent,
            bucket_config=bucket_config,
            crop_record=crop_record,
        )
    return CropMeta(
        crop_center_x=cx,
        crop_center_y=cy,
        fitted_width=fitted.width,
        fitted_height=fitted.height,
        source_width=source_width,
        source_height=source_height,
        state=state,
        enable_bucket=False,
    )


def bake_image_to_prepared(
    *,
    source_path: Path,
    prepared_dir: Path,
    bucket_config: BucketPreprocessConfig,
    center_x: float,
    center_y: float,
    stored: StoredCropRecord | None = None,
) -> tuple[Path, BucketAssignment | None]:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    image = _load_rgb(source_path)
    bucket_assignment: BucketAssignment | None = None
    if bucket_config.enable_bucket:
        bucket_assignment = _resolve_assignment(
            source_width=image.width,
            source_height=image.height,
            bucket_config=bucket_config,
            center_x=center_x,
            center_y=center_y,
            stored=stored,
        )
        result = apply_bucket_crop(image, bucket_assignment)
    else:
        result = apply_crop(image, bucket_config.resolution, center_x, center_y)
    output_path = prepared_jpg_path(prepared_dir, source_path.name)
    result.save(output_path, format="JPEG", quality=95)
    _cleanup_stale_prepared(prepared_dir, source_path.stem, output_path)
    return output_path, bucket_assignment


def build_fitted_preview_bytes(source_path: Path, bucket_config: BucketPreprocessConfig) -> bytes:
    from io import BytesIO

    image = _load_rgb(source_path)
    if bucket_config.enable_bucket:
        assignment = assign_bucket(
            image.width,
            image.height,
            resolution=bucket_config.resolution,
            min_bucket_reso=bucket_config.min_bucket_reso,
            max_bucket_reso=bucket_config.max_bucket_reso,
            bucket_reso_steps=bucket_config.bucket_reso_steps,
            bucket_no_upscale=bucket_config.bucket_no_upscale,
        )
        preview = image.resize(
            (assignment.scale_to_width, assignment.scale_to_height),
            Image.Resampling.LANCZOS,
        )
    else:
        preview = _resize_to_fit(image, bucket_config.resolution)
    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def compute_preprocess_status(
    dataset: Dataset,
    crop_by_filename: dict[str, StoredCropRecord],
) -> PreprocessStatus:
    image_dir = Path(dataset.image_dir)
    filenames = list_image_filenames(image_dir)
    bucket_config = BucketPreprocessConfig.from_dataset(dataset)
    counts = {state: 0 for state in ImagePreprocessState}
    for filename in filenames:
        crop_data = crop_by_filename.get(filename)
        if crop_data is None or bucket_config is None:
            state = ImagePreprocessState.NO_CROP
        else:
            state = get_image_state(
                filename=filename,
                image_dir=image_dir,
                bucket_config=bucket_config,
                crop_record=crop_data,
            )
        counts[state] += 1
    return PreprocessStatus(
        target_resolution=dataset.target_resolution,
        preprocess_ready=dataset.preprocess_ready,
        total=len(filenames),
        no_crop=counts[ImagePreprocessState.NO_CROP],
        stale=counts[ImagePreprocessState.STALE],
        cropped=counts[ImagePreprocessState.CROPPED],
        ready=counts[ImagePreprocessState.READY],
    )


def recompute_preprocess_ready(
    dataset: Dataset,
    crop_by_filename: dict[str, StoredCropRecord],
) -> bool:
    status = compute_preprocess_status(dataset, crop_by_filename)
    if status.total == 0 or dataset.target_resolution is None:
        return False
    return status.ready == status.total


def invalidate_latent_cache_for_prepared(prepared_path: Path) -> None:
    npz_path = prepared_path.parent / f"{prepared_path.stem}_sdxl.npz"
    if npz_path.is_file():
        npz_path.unlink(missing_ok=True)
