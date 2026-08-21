import type { ParameterMeta } from "../parameterUtils";

export const optimizationParameterMetadata: ParameterMeta[] = [
// Optimization
  {
    key: "mixed_precision",
    label: "Mixed Precision",
    section: "Optimization",
    shortHint: "Training compute precision; float16/bfloat16 reduce VRAM.",
    description:
      "Global mixed precision mode for training computations. float16 is widely supported; bfloat16 offers better numerical range on Ampere+ GPUs. Affects speed and memory across UNet and optimizer states.",
    defaultValue: "float16",
    constraints: "float32 | float16 | bfloat16",
    recommendedValue: "float16",
    valueOptions: [
      { value: "float32", description: "No mixed precision; highest memory use, maximum numerical precision." },
      { value: "float16", description: "Standard half-precision training; widely supported and VRAM-efficient." },
      { value: "bfloat16", description: "Brain float mixed precision; better range than float16 on Ampere+ GPUs." },
    ],
  },
  {
    key: "seed",
    label: "Seed",
    section: "Optimization",
    shortHint: "Random seed for reproducibility; leave empty for random.",
    description:
      "Fixes random number generation for dataset shuffling, noise sampling, and weight initialization. Set a specific integer to reproduce a training run. Omit or leave empty for a random seed each run.",
    defaultValue: "random",
    showInlineHint: false,
    recommendedValue: "random",
  },
  {
    key: "gradient_checkpointing",
    label: "Gradient Checkpointing",
    section: "Optimization",
    shortHint: "Trade compute for VRAM by recomputing activations during backward pass.",
    description:
      "Recomputes intermediate activations during backprop instead of storing them, significantly reducing VRAM at the cost of ~20–30% slower training. Strongly recommended for SDXL LoRA on consumer GPUs.",
    defaultValue: "true",
    showInlineHint: false,
    recommendedValue: "true",
    valueOptions: [
      { value: "true", description: "Trade ~20–30% speed for significant VRAM savings." },
      { value: "false", description: "Store all activations; faster but uses more VRAM." },
    ],
  },
];
