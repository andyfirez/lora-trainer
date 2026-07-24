# ComfyUI 0.27.0 (vendored)

This directory contains a vendored copy of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) **0.27.0**
(GPL-3.0) used for SDXL sampling inference only.

- Source pin: ComfyUI 0.27.0
- Copied from upstream `comfy/` package (minus unused audio/background_removal/cldm)
- Includes `node_helpers.py` from ComfyUI root (required by `comfy/hooks.py`)
- Loaded via `src/trainer/sdxl/comfy_inference/bootstrap.py` — **no external ComfyUI install required**
- Optional override: `LORA_TRAINER_COMFY_VENDOR` env var

Training continues to use HuggingFace diffusers + PEFT.

See `THIRD_PARTY_COMFYUI.md` at the repository root for attribution details.
