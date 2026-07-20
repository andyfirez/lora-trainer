"""Tests for Pillow grid compositor."""

from pathlib import Path

from PIL import Image

from src.sampler.sweep.grid_compositor import compose_grid


def test_compose_grid_creates_file(tmp_path: Path) -> None:
    cell = tmp_path / "cell.png"
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(cell)
    output = tmp_path / "grid.png"
    compose_grid(
        [[cell, cell], [cell, cell]],
        x_axis="prompt",
        y_axis="lora_weight",
        x_values=["a", "b"],
        y_values=[0.5, 1.0],
        title="test grid",
        output_path=output,
        cell_size=64,
    )
    assert output.is_file()
    with Image.open(output) as img:
        assert img.width > 64
        assert img.height > 64
