"""Read generation metadata embedded in image files."""

from __future__ import annotations

from PIL import Image

IGNORED_INFO_KEYS = frozenset(
    {
        "jfif",
        "jfif_version",
        "jfif_unit",
        "jfif_density",
        "dpi",
        "exif",
        "loop",
        "background",
        "timestamp",
        "duration",
        "progressive",
        "progression",
        "icc_profile",
        "chromaticity",
        "photoshop",
    }
)


def _normalize_info_value(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def read_info_from_image(image: Image.Image) -> tuple[str | None, dict[str, str]]:
    """Return generation info text and remaining metadata key/value pairs."""
    items = {key: _normalize_info_value(value) for key, value in (image.info or {}).items()}

    geninfo = items.pop("parameters", None)

    if geninfo is None and "comment" in items:
        geninfo = items.pop("comment")

    for field in IGNORED_INFO_KEYS:
        items.pop(field, None)

    return geninfo, items
