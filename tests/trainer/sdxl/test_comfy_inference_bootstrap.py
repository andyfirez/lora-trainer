"""Tests for vendored Comfy inference bootstrap."""

from pathlib import Path

from src.trainer.sdxl.comfy_inference.bootstrap import resolve_vendor_root


def test_resolve_vendor_root_points_at_repo_vendor_tree() -> None:
    root = resolve_vendor_root()
    assert (root / "comfy" / "sd.py").is_file()
    assert "comfyui-0.27.0" in str(root)


def test_vendor_contains_gpl_license() -> None:
    root = resolve_vendor_root()
    license_path = root / "LICENSE"
    assert license_path.is_file() or (root.parent.parent / "LICENSE").exists()


def test_vendored_comfy_imports_after_bootstrap() -> None:
    from src.trainer.sdxl.comfy_inference.bootstrap import ensure_vendored_comfy

    ensure_vendored_comfy()
    import comfy.diffusers_load  # noqa: F401
    import comfy.sd  # noqa: F401
