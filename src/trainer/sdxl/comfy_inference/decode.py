"""Convert vendored Comfy VAE output to PIL."""

from __future__ import annotations

import numpy as np
from PIL import Image
from torch import Tensor


def comfy_tensor_to_pil(image: Tensor) -> Image.Image:
    if image.ndim == 4:
        image = image[0]
    array = (image.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)
