import type { ParameterMeta } from "../parameterUtils";
import { diffusersSchedulerOptions } from "../sampleSamplerOptions";

const SAMPLE_SCHEDULER_DESCRIPTIONS: Record<string, string> = {
  euler: "Euler sampler; fast and matches most SDXL workflows.",
  euler_a: "Euler ancestral; adds stochasticity for more varied previews.",
  ddim: "DDIM deterministic sampler; reproducible previews.",
  "dpm++": "DPM++ solver; higher quality previews at the cost of slower sampling.",
};

const sampleSchedulerValueOptions = diffusersSchedulerOptions.map((option) => ({
  value: option.value,
  description: SAMPLE_SCHEDULER_DESCRIPTIONS[option.value] ?? option.label,
}));

export const samplingParameterMetadata: ParameterMeta[] = [
// Runtime sampling overlay (yaml-only; populated during sampling jobs)
  {
    key: "sample_prompts",
    label: "Sample Prompts",
    section: "Sampling",
    shortHint: "Prompts used by standalone sampling jobs (resolved from sampling config).",
    description:
      "List of text prompts for sampling jobs. In the web UI these come from the linked sampling config via resolve_sampling(); inline YAML values are merged when creating sampling jobs.",
    defaultValue: "[]",
    yamlOnly: true,
    recommendedValue: "[]",
  },
  {
    key: "sample_negative_prompt",
    label: "Sample Negative Prompt",
    section: "Sampling",
    shortHint: "Negative prompt for post-train sampling jobs.",
    description:
      "Negative prompt text applied during post-train sampling. Typically resolved from the linked sampling config rather than set inline in training YAML.",
    defaultValue: '""',
    yamlOnly: true,
    recommendedValue: '""',
  },
  {
    key: "sample_steps",
    label: "Sample Steps",
    section: "Sampling",
    shortHint: "Number of denoising steps for preview images.",
    description:
      "Inference steps for post-train sampling jobs. Higher values improve output quality at the cost of slower generation. Resolved from the linked sampling config in normal workflows.",
    defaultValue: "30",
    constraints: "≥ 1",
    yamlOnly: true,
    recommendedValue: "30",
    rangeGuidance: [
      { range: "15–20", description: "Fast previews; lower quality but quick feedback." },
      { range: "30", description: "Balanced preview quality and speed." },
      { range: "40–50", description: "Higher quality previews; slower after each checkpoint." },
    ],
  },
  {
    key: "sample_cfg_scale",
    label: "Sample CFG Scale",
    section: "Sampling",
    shortHint: "Classifier-free guidance scale for preview images.",
    description:
      "CFG scale controlling prompt adherence during post-train sampling. SDXL commonly uses 5–8. Resolved from the linked sampling config in normal workflows.",
    defaultValue: "7.5",
    constraints: "> 0",
    yamlOnly: true,
    recommendedValue: "7.5",
    rangeGuidance: [
      { range: "5–6", description: "Softer prompt adherence; more natural previews." },
      { range: "7–8", description: "SDXL common range; good default for previews." },
      { range: "10+", description: "Strong guidance; may look oversaturated or artifact-heavy." },
    ],
  },
  {
    key: "sample_width",
    label: "Sample Width",
    section: "Sampling",
    shortHint: "Output width for preview images; null uses training resolution.",
    description:
      "Width in pixels for post-train sampling output. When null, the sampler uses the training resolution. Resolved from the linked sampling config in normal workflows.",
    defaultValue: "null (uses resolution)",
    constraints: "64–2048",
    yamlOnly: true,
    recommendedValue: "null (uses resolution)",
    rangeGuidance: [
      { range: "null", description: "Uses training resolution; simplest and consistent." },
      { range: "1024", description: "Standard SDXL preview width." },
      { range: "1280–1536", description: "Wider previews; more VRAM during sampling." },
    ],
  },
  {
    key: "sample_height",
    label: "Sample Height",
    section: "Sampling",
    shortHint: "Output height for preview images; null uses training resolution.",
    description:
      "Height in pixels for post-train sampling output. When null, the sampler uses the training resolution. Resolved from the linked sampling config in normal workflows.",
    defaultValue: "null (uses resolution)",
    constraints: "64–2048",
    yamlOnly: true,
    recommendedValue: "null (uses resolution)",
    rangeGuidance: [
      { range: "null", description: "Uses training resolution; simplest and consistent." },
      { range: "1024", description: "Standard SDXL preview height." },
      { range: "1280–1536", description: "Taller previews; more VRAM during sampling." },
    ],
  },
  {
    key: "sample_scheduler",
    label: "Sample Scheduler",
    section: "Sampling",
    shortHint: "Noise scheduler for preview image generation.",
    description:
      "Sampler algorithm for post-train sampling jobs. euler is the default and matches most SDXL workflows. Resolved from the linked sampling config in normal workflows.",
    defaultValue: "euler",
    constraints: "euler | euler_a | ddim | dpm++",
    yamlOnly: true,
    recommendedValue: "euler",
    valueOptions: sampleSchedulerValueOptions,
  },
  {
    key: "sample_vae_tiling",
    label: "Sample VAE Tiling",
    section: "Sampling",
    shortHint: "Tile VAE decode to reduce VRAM during preview generation.",
    description:
      "Enables tiled VAE decoding during post-train sampling, trading a small speed penalty for lower peak VRAM usage when generating large images.",
    defaultValue: "true",
    recommendedValue: "true",
    valueOptions: [
      { value: "true", description: "Tile VAE decode to reduce peak VRAM." },
      { value: "false", description: "Full VAE decode; faster but more VRAM." },
    ],
  },
  {
    key: "sample_vae_fp32",
    label: "Sample VAE FP32",
    section: "Sampling",
    shortHint: "Run VAE decode in float32 for higher preview fidelity.",
    description:
      "Forces float32 precision during VAE decode in sampling. Can reduce color banding at the cost of extra VRAM and slower decode.",
    defaultValue: "false",
    yamlOnly: true,
    recommendedValue: "false",
    valueOptions: [
      { value: "true", description: "Higher fidelity decode; more VRAM and slower." },
      { value: "false", description: "Use configured VAE precision." },
    ],
  },
  {
    key: "sample_offload_unet_before_decode",
    label: "Sample Offload UNet Before Decode",
    section: "Sampling",
    shortHint: "Move UNet off GPU before VAE decode to free VRAM.",
    description:
      "Offloads the UNet from GPU memory before VAE decoding during post-train sampling. Helps avoid OOM on consumer GPUs when generating images at full resolution.",
    defaultValue: "true",
    yamlOnly: true,
    recommendedValue: "true",
    valueOptions: [
      { value: "true", description: "Free GPU memory before VAE decode; helps avoid OOM." },
      { value: "false", description: "Keep UNet on GPU during decode." },
    ],
  },
  {
    key: "post_training_sampling_config_id",
    label: "Post-Training Sampling Config",
    section: "Sampling",
    shortHint: "Deprecated — rejected by config validation.",
    description:
      "Legacy field for linking a sampling config after training completes. Rejected by config validation. Do not use in new configs.",
    yamlOnly: true,
    deprecated: true,
    showInlineHint: false,
    recommendedValue: "do not use",
  },
  {
    key: "sample_after_training",
    label: "Sample After Training",
    section: "Sampling",
    shortHint: "Deprecated — rejected by config validation.",
    description:
      "Deprecated boolean flag for post-training sampling. Config validation rejects this key.",
    yamlOnly: true,
    deprecated: true,
    recommendedValue: "do not use",
  },
  {
    key: "sampling_enabled",
    label: "Sampling Enabled",
    section: "Sampling",
    shortHint: "Deprecated — rejected by config validation.",
    description: "Legacy post-training auto-sampling flag. Rejected by config validation.",
    yamlOnly: true,
    deprecated: true,
    showInlineHint: false,
    recommendedValue: "do not use",
  },
  {
    key: "sampling_config_id",
    label: "Sampling Config ID",
    section: "Sampling",
    shortHint: "Deprecated — rejected by config validation.",
    description: "Legacy reference to a sampling config for post-training jobs. Rejected by config validation.",
    yamlOnly: true,
    deprecated: true,
    showInlineHint: false,
    recommendedValue: "do not use",
  },
];
