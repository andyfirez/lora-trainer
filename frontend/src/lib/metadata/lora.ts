import type { ParameterMeta } from "../parameterUtils";

export const loraParameterMetadata: ParameterMeta[] = [
// LoRA
  {
    key: "lora_rank",
    label: "Rank",
    section: "LoRA",
    shortHint: "Adapter capacity — higher rank learns more detail but uses more VRAM and risks overfitting.",
    description:
      "Rank (dimension) of the low-rank adaptation matrices. Higher values increase expressiveness and VRAM usage. Typical values: 8–32 for style/subject LoRAs, 64–128 for complex concepts. Pair with lora_alpha for effective learning strength.",
    defaultValue: "32",
    constraints: "1–256",
    recommendedValue: "32",
    rangeGuidance: [
      { range: "8–16", description: "Light style or simple subject LoRAs; fast training, low VRAM." },
      { range: "32", description: "General-purpose default for most SDXL character and style LoRAs." },
      { range: "64–128", description: "Complex concepts or fine details; higher VRAM and overfitting risk." },
    ],
  },
  {
    key: "lora_alpha",
    label: "Alpha",
    section: "LoRA",
    shortHint: "Scaling factor for LoRA weights; often set equal to rank for unit scaling.",
    description:
      "Alpha scales the LoRA contribution at inference time (effective scale ≈ alpha / rank). Setting alpha equal to rank is a common default. Lower alpha softens the effect; higher alpha amplifies learned features but can cause artifacts.",
    defaultValue: "32.0",
    constraints: "> 0",
    recommendedValue: "32.0",
    rangeGuidance: [
      { range: "= rank", description: "Unit scaling (alpha/rank = 1); most common community default." },
      { range: "rank × 0.5", description: "Softer LoRA effect at inference; reduces artifacts." },
      { range: "rank × 2", description: "Stronger effect; may cause oversaturation or artifacts." },
    ],
  },
  {
    key: "lora_dropout",
    label: "Dropout",
    section: "LoRA",
    shortHint: "Regularization on LoRA layers; 0 disables dropout.",
    description:
      "Dropout probability applied to LoRA adapter layers during training. Can reduce overfitting on small datasets. Most SDXL LoRA recipes use 0.0; try 0.05–0.1 only if you see memorization on tiny datasets.",
    defaultValue: "0.0",
    constraints: "0.0–0.999",
    recommendedValue: "0.0",
    rangeGuidance: [
      { range: "0.0", description: "No dropout; standard for most SDXL LoRA recipes." },
      { range: "0.05–0.1", description: "Mild regularization for very small datasets prone to memorization." },
    ],
  },
];
