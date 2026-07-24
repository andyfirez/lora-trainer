# ComfyUI Sampling Attribution

Portions of the SDXL sampling backend under `src/trainer/sdxl/latent_sampling/comfy/` are
adapted from [ComfyUI](https://github.com/comfyanonymous/ComfyUI) version **0.27.0**, which is
licensed under the **GNU General Public License v3.0**.

Primary source files:

- `comfy/model_sampling.py` — EPS sigma model and noise scaling
- `comfy/samplers.py` — sigma scheduler dispatch (`simple`, `karras`)
- `comfy/k_diffusion/sampling.py` — sampler step functions (`euler`, `euler_ancestral`, `dpmpp_2m`)
- `comfy/sample.py` — CPU-seeded noise preparation

Behavior in LoRA Trainer is pinned to ComfyUI 0.27.0 for the supported V1 pairs:

- `euler` + `simple`
- `euler_ancestral` + `simple`
- `dpmpp_2m` + `karras`

The adapted code is integrated with the existing diffusers-based UNet/VAE/CLIP loading path.
Full model-stack parity is tracked separately in Beads issue `lora-trainer-gj1.9` if manual
visual review shows sampler-only parity is insufficient.
