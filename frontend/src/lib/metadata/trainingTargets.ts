import type { ParameterMeta } from "../parameterUtils";

export const trainingTargetsParameterMetadata: ParameterMeta[] = [
// Training Targets
  {
    key: "unet.train",
    label: "UNet — Train",
    section: "Training Targets",
    shortHint: "Whether to train LoRA weights on the UNet (image denoiser). Usually enabled.",
    description:
      "When enabled, LoRA adapters are attached to UNet attention and feed-forward layers — this is the primary target for visual learning. Disabling UNet training while training text encoders is a niche setup for text-only fine-tuning.",
    defaultValue: "true",
    showInlineHint: false,
    recommendedValue: "true",
    valueOptions: [
      { value: "true", description: "Train LoRA on UNet layers — primary target for visual learning." },
      { value: "false", description: "Skip UNet training; niche text-only fine-tuning setup." },
    ],
  },
  {
    key: "unet.weight_dtype",
    label: "UNet — Weight Dtype",
    section: "Training Targets",
    shortHint: "Precision for UNet weights during training.",
    description:
      "Floating-point dtype for UNet parameters. float16 and bfloat16 reduce VRAM; bfloat16 is often more stable on Ampere+ GPUs. float32 uses the most memory but can help with numerical instability on older hardware.",
    defaultValue: "float16",
    constraints: "float32 | float16 | bfloat16",
    recommendedValue: "float16",
    valueOptions: [
      { value: "float32", description: "Full precision; highest VRAM use, best numerical stability on older GPUs." },
      { value: "float16", description: "Half precision; good balance of speed and memory on most GPUs." },
      { value: "bfloat16", description: "Brain float; wider dynamic range than float16, preferred on Ampere+ GPUs." },
    ],
  },
  {
    key: "unet.learning_rate",
    label: "UNet — Learning Rate",
    section: "Training Targets",
    shortHint: "Learning rate for UNet LoRA weights.",
    description:
      "Step size for UNet weight updates. Typical SDXL LoRA range: 1e-4 to 5e-4 for AdamW. Only applies when UNet training is enabled.",
    defaultValue: "5e-5",
    constraints: "> 0",
    recommendedValue: "5e-5",
    rangeGuidance: [
      { range: "1e-5", description: "Conservative; slower convergence." },
      { range: "5e-5", description: "Common default for SDXL LoRA UNet training." },
      { range: "1e-4 – 5e-4", description: "Aggressive; faster training, often used for UNet-only runs." },
    ],
  },
  {
    key: "text_encoder_1.train",
    label: "Text Encoder 1 — Train",
    section: "Training Targets",
    shortHint: "Train LoRA on CLIP-L text encoder. Increases VRAM and training time.",
    description:
      "Enables LoRA fine-tuning on the first SDXL text encoder (CLIP ViT-L). Useful when trigger words or captions need stronger semantic binding. Incompatible with caching text encoder outputs while training TEs.",
    defaultValue: "false",
    showInlineHint: false,
    recommendedValue: "false",
    valueOptions: [
      { value: "true", description: "Train CLIP-L; stronger trigger-word binding, more VRAM." },
      { value: "false", description: "Frozen text encoder 1; standard for most LoRAs." },
    ],
  },
  {
    key: "text_encoder_1.weight_dtype",
    label: "Text Encoder 1 — Weight Dtype",
    section: "Training Targets",
    shortHint: "Precision for text encoder 1 weights.",
    description:
      "Floating-point dtype for text encoder 1 parameters. Same trade-offs as UNet dtype — lower precision saves VRAM with minimal quality impact for most LoRA runs.",
    defaultValue: "float16",
    constraints: "float32 | float16 | bfloat16",
    recommendedValue: "float16",
    valueOptions: [
      { value: "float32", description: "Full precision for text encoder 1 weights." },
      { value: "float16", description: "Half precision; saves VRAM with minimal quality impact." },
      { value: "bfloat16", description: "Brain float; stable on Ampere+ GPUs." },
    ],
  },
  {
    key: "text_encoder_1.learning_rate",
    label: "Text Encoder 1 — Learning Rate",
    section: "Training Targets",
    shortHint: "Learning rate for CLIP-L LoRA weights.",
    description:
      "Step size for text encoder 1 weight updates. Typically lower than UNet (e.g. 5e-5). Only applies when text encoder 1 training is enabled.",
    defaultValue: "5e-5",
    constraints: "> 0",
    recommendedValue: "5e-5",
    rangeGuidance: [
      { range: "1e-5", description: "Conservative text encoder LR." },
      { range: "5e-5", description: "Typical text encoder LR for SDXL LoRA." },
    ],
  },
  {
    key: "text_encoder_2.train",
    label: "Text Encoder 2 — Train",
    section: "Training Targets",
    shortHint: "Train LoRA on OpenCLIP-G text encoder. Rarely needed for most LoRAs.",
    description:
      "Enables LoRA fine-tuning on SDXL's second text encoder (OpenCLIP-G). Typically left disabled unless you need fine-grained caption semantics. Significantly increases memory when combined with UNet training.",
    defaultValue: "false",
    showInlineHint: false,
    recommendedValue: "false",
    valueOptions: [
      { value: "true", description: "Train OpenCLIP-G; rare, increases memory significantly." },
      { value: "false", description: "Frozen text encoder 2; recommended default." },
    ],
  },
  {
    key: "text_encoder_2.weight_dtype",
    label: "Text Encoder 2 — Weight Dtype",
    section: "Training Targets",
    shortHint: "Precision for text encoder 2 weights.",
    description:
      "Floating-point dtype for text encoder 2 parameters.",
    defaultValue: "float16",
    constraints: "float32 | float16 | bfloat16",
    recommendedValue: "float16",
    valueOptions: [
      { value: "float32", description: "Full precision for text encoder 2 weights." },
      { value: "float16", description: "Half precision; saves VRAM with minimal quality impact." },
      { value: "bfloat16", description: "Brain float; stable on Ampere+ GPUs." },
    ],
  },
  {
    key: "text_encoder_2.learning_rate",
    label: "Text Encoder 2 — Learning Rate",
    section: "Training Targets",
    shortHint: "Learning rate for OpenCLIP-G LoRA weights.",
    description:
      "Step size for text encoder 2 weight updates. Only applies when text encoder 2 training is enabled.",
    defaultValue: "5e-5",
    constraints: "> 0",
    recommendedValue: "5e-5",
    rangeGuidance: [
      { range: "1e-5", description: "Conservative text encoder LR." },
      { range: "5e-5", description: "Typical text encoder LR for SDXL LoRA." },
    ],
  },
  {
    key: "clip_skip",
    label: "CLIP Skip",
    section: "Training Targets",
    shortHint: "Which CLIP hidden layer encodes text; 2 matches Kohya SDXL defaults.",
    description:
      "Number of layers to skip from the end of the CLIP text encoder when producing embeddings. Value 2 is the Kohya SDXL default and matches most community models. Changing this alters how captions influence generation — keep consistent between training and inference.",
    defaultValue: "2",
    constraints: "≥ 1",
    recommendedValue: "2",
    rangeGuidance: [
      { range: "2", description: "Kohya SDXL default; matches most community models and workflows." },
      { range: "1", description: "No skip; different text embedding distribution, rarely used for SDXL." },
      { range: "3+", description: "Deeper skip; alters caption influence significantly — keep consistent at inference." },
    ],
  },
];
