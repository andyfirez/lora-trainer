# ComfyUI Sampling Attribution

Portions of the SDXL **sampling** backend are adapted from [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
version **0.27.0**, licensed under the **GNU General Public License v3.0**.

## Vendored inference stack (primary)

The full SDXL txt2img inference path uses ComfyUI logic **vendored in the repository** at
`vendor/comfyui-0.27.0/`. Integration glue lives in `src/trainer/sdxl/comfy_inference/`.

No external ComfyUI installation is required. Bootstrap loads `vendor/comfyui-0.27.0` via
`ensure_vendored_comfy()` (optional override: `LORA_TRAINER_COMFY_VENDOR`).

Primary upstream sources:

- `comfy/sd.py` — checkpoint/diffusers loading, CLIP, VAE, LoRA
- `comfy/model_base.py` — SDXL UNet forward, `encode_adm`
- `comfy/latent_formats.py` — SDXL latent scale (0.13025)
- `comfy/sample.py`, `comfy/samplers.py`, `comfy/k_diffusion/sampling.py` — KSampler loop
- `comfy/sdxl_clip.py` — dual CLIP encoding
- `comfy/lora.py`, `comfy/weight_adapter/` — LoRA patch application

**Training** continues to use HuggingFace diffusers + PEFT and is not affected.

## Reimplemented sampler helpers (legacy bridge)

Under `src/trainer/sdxl/latent_sampling/comfy/` — early independent ports of sigma/sampler math
used during the sampler-only iteration. Production sampling uses the vendored stack above.

## Version pin

Behavior is pinned to ComfyUI **0.27.0**. Supported V1 sampler pairs:

- `euler` + `simple`
- `euler_ancestral` + `simple`
- `dpmpp_2m` + `karras`

Manual visual review against ComfyUI 0.27.0 is the acceptance gate for parity.
