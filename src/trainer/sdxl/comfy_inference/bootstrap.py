"""Bootstrap vendored ComfyUI 0.27.0 package from vendor/comfyui-0.27.0."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

_VENDOR_ROOT = Path(__file__).resolve().parents[4] / "vendor" / "comfyui-0.27.0"


def resolve_vendor_root() -> Path:
    override = os.environ.get("LORA_TRAINER_COMFY_VENDOR", "").strip()
    root = Path(override).expanduser().resolve() if override else _VENDOR_ROOT.resolve()
    if not (root / "comfy" / "sd.py").is_file():
        raise RuntimeError(
            f"Vendored ComfyUI tree missing at {root}. "
            "Expected vendor/comfyui-0.27.0/comfy/sd.py in the repository."
        )
    return root


@lru_cache(maxsize=1)
def ensure_vendored_comfy() -> Path:
    if "--disable-dynamic-vram" not in sys.argv:
        sys.argv.append("--disable-dynamic-vram")

    root = resolve_vendor_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    import comfy.utils

    comfy.utils.PROGRESS_BAR_ENABLED = False
    return root
