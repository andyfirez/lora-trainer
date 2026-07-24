# Manual visual review checklist (ComfyUI 0.27.0 parity)

Use the same base checkpoint, LoRA (if any), prompt, negative prompt, seed, steps,
CFG, width/height, and sampler pair in both LoRA Trainer and ComfyUI 0.27.0.

## Sampler pairs (V1 scope)

1. `euler` + `simple`
2. `euler_ancestral` + `simple`
3. `dpmpp_2m` + `karras`

## Compare

- Base model txt2img (no LoRA)
- Trained LoRA at weight 1.0
- Optional: bucket-aligned `add_time_ids` when training used buckets

## Pass criteria

Qualitative visual similarity (user judgment). No automated pixel diff required.

Record outcome in beads issue `lora-trainer-gj1` before closing the epic.
