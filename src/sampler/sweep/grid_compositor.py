"""Compose labeled grid PNG images with Pillow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

CELL_SIZE = 512
LABEL_HEIGHT = 44
LABEL_WIDTH = 160
PADDING = 8
TITLE_HEIGHT = 52
LABEL_FONT_SIZE = 18
TITLE_FONT_SIZE = 22
BG_COLOR = (24, 24, 28)
TEXT_COLOR = (220, 220, 225)
BORDER_COLOR = (60, 60, 70)


def _load_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _truncate(text: str, max_len: int = 48) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_label(key: str, value: Any) -> str:
    if value is None:
        return "base model" if key == "lora_path" else "—"
    if key == "lora_path" and isinstance(value, str):
        if value == "base model" or " (" in value:
            return _truncate(value)
        return _truncate(Path(value).stem)
    if isinstance(value, float):
        return f"{value:g}"
    return _truncate(str(value))


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = left + (right - left - text_width) // 2
    y = top + (bottom - top - text_height) // 2
    draw.text((x, y), text, fill=TEXT_COLOR, font=font)


def compose_grid(
    cell_paths: list[list[Path | None]],
    *,
    x_axis: str,
    y_axis: str,
    x_values: list[Any],
    y_values: list[Any],
    title: str = "",
    output_path: Path,
    cell_size: int = CELL_SIZE,
) -> Path:
    n_cols = len(x_values)
    n_rows = len(y_values)
    width = LABEL_WIDTH + PADDING + n_cols * (cell_size + PADDING) + PADDING
    height = (TITLE_HEIGHT if title else 0) + LABEL_HEIGHT + PADDING + n_rows * (cell_size + PADDING) + PADDING
    canvas = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    font = _load_font(LABEL_FONT_SIZE)
    title_font = _load_font(TITLE_FONT_SIZE)

    y_offset = PADDING
    if title:
        _draw_centered_text(draw, title, (PADDING, y_offset, width - PADDING, y_offset + TITLE_HEIGHT), title_font)
        y_offset += TITLE_HEIGHT

    header_y = y_offset
    for col, x_val in enumerate(x_values):
        x = LABEL_WIDTH + PADDING + col * (cell_size + PADDING)
        label = _format_label(x_axis, x_val)
        _draw_centered_text(
            draw,
            label,
            (x, header_y, x + cell_size, header_y + LABEL_HEIGHT),
            font,
        )

    grid_top = header_y + LABEL_HEIGHT
    for row, y_val in enumerate(y_values):
        y = grid_top + row * (cell_size + PADDING)
        label = _format_label(y_axis, y_val)
        _draw_centered_text(
            draw,
            label,
            (PADDING, y, PADDING + LABEL_WIDTH, y + cell_size),
            font,
        )
        row_paths = cell_paths[row] if row < len(cell_paths) else []
        for col in range(n_cols):
            x = LABEL_WIDTH + PADDING + col * (cell_size + PADDING)
            path = row_paths[col] if col < len(row_paths) else None
            box = (x, y, x + cell_size, y + cell_size)
            draw.rectangle(box, outline=BORDER_COLOR)
            if path is not None and path.is_file():
                with Image.open(path) as img:
                    thumb = img.convert("RGB")
                    thumb.thumbnail((cell_size - 2, cell_size - 2), Image.Resampling.LANCZOS)
                    tx = x + (cell_size - thumb.width) // 2
                    ty = y + (cell_size - thumb.height) // 2
                    canvas.paste(thumb, (tx, ty))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path
