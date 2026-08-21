import type { ParameterMeta } from "../parameterUtils";

export const performanceParameterMetadata: ParameterMeta[] = [
// Performance
  {
    key: "cache_latents",
    label: "Cache Latents (RAM)",
    section: "Performance",
    shortHint: "Pre-encode images to VAE latents in RAM — major speedup, uses system memory.",
    description:
      "Encodes all training images to VAE latents once and keeps them in RAM, skipping VAE encoding each step. Dramatically speeds training but requires sufficient system RAM proportional to dataset size.",
    defaultValue: "true",
    recommendedValue: "true",
    valueOptions: [
      { value: "true", description: "Pre-encode to RAM; major speedup, needs system memory." },
      { value: "false", description: "Encode images each step; slower but lower RAM use." },
    ],
  },
  {
    key: "cache_latents_to_disk",
    label: "Cache Latents to Disk",
    section: "Performance",
    shortHint: "Persist VAE latents as .npz files; requires RAM caching enabled.",
    description:
      "Saves encoded latents to disk (.npz) so subsequent runs skip VAE encoding entirely. Requires cache_latents enabled. Useful for iterative hyperparameter tuning on the same dataset.",
    defaultValue: "false",
    recommendedValue: "false",
    valueOptions: [
      { value: "true", description: "Persist latents to disk for reuse across runs." },
      { value: "false", description: "RAM-only latent cache." },
    ],
  },
  {
    key: "cache_text_encoder_outputs",
    label: "Cache Text Encoder Outputs (RAM)",
    section: "Performance",
    shortHint: "Pre-compute text embeddings in RAM. Incompatible with training text encoders.",
    description:
      "Caches CLIP text encoder outputs for all captions in RAM, skipping text encoding each step. Cannot be used while training text encoders (text_encoder_1.train or text_encoder_2.train is true).",
    defaultValue: "true",
    recommendedValue: "true",
    valueOptions: [
      { value: "true", description: "Pre-compute text embeddings in RAM; incompatible with TE training." },
      { value: "false", description: "Encode captions each step." },
    ],
  },
  {
    key: "cache_text_encoder_outputs_to_disk",
    label: "Cache Text Encoder Outputs to Disk",
    section: "Performance",
    shortHint: "Persist text embeddings to disk; requires RAM TE caching.",
    description:
      "Saves text encoder outputs to disk for reuse across runs. Requires cache_text_encoder_outputs enabled.",
    defaultValue: "false",
    recommendedValue: "false",
    valueOptions: [
      { value: "true", description: "Persist text embeddings to disk." },
      { value: "false", description: "RAM-only TE output cache." },
    ],
  },
  {
    key: "attention_mechanism",
    label: "Attention Mechanism",
    section: "Performance",
    shortHint: "Attention kernel backend; sdpa is the PyTorch 2.x default.",
    description:
      "Selects the attention implementation. sdpa uses PyTorch scaled dot-product attention (fast, no extra deps). xformers can be faster on some GPUs but requires the xformers package. default uses diffusers' built-in attention.",
    defaultValue: "sdpa",
    constraints: "default | sdpa | xformers",
    recommendedValue: "sdpa",
    valueOptions: [
      { value: "default", description: "Diffusers built-in attention; no extra dependencies." },
      { value: "sdpa", description: "PyTorch scaled dot-product attention; fast default on PyTorch 2.x." },
      { value: "xformers", description: "xFormers memory-efficient attention; can be faster on some GPUs." },
    ],
  },
  {
    key: "vae_dtype",
    label: "VAE Dtype",
    section: "Performance",
    shortHint: "Precision for VAE encode/decode operations.",
    description:
      "Data type for VAE operations during training and sampling. auto selects based on GPU capability. Lower precision saves VRAM during latent caching and sampling decode.",
    defaultValue: "auto",
    constraints: "auto | float32 | float16 | bfloat16",
    recommendedValue: "auto",
    valueOptions: [
      { value: "auto", description: "Automatically select based on GPU capability." },
      { value: "float32", description: "Full precision VAE; highest fidelity, most VRAM." },
      { value: "float16", description: "Half precision VAE; saves VRAM during encode/decode." },
      { value: "bfloat16", description: "Brain float VAE; good range on Ampere+ GPUs." },
    ],
  },
  {
    key: "tf32",
    label: "TF32",
    section: "Performance",
    shortHint: "Use TensorFloat-32 on Ampere+ GPUs for faster matmuls.",
    description:
      "Enables TF32 mode for matrix multiplications on NVIDIA Ampere and newer GPUs. Provides a free speed boost with negligible quality impact for most LoRA training.",
    defaultValue: "true",
    recommendedValue: "true",
    valueOptions: [
      { value: "true", description: "Free speed boost on Ampere+ GPUs with negligible quality impact." },
      { value: "false", description: "Disable TF32 matmul acceleration." },
    ],
  },
  {
    key: "torch_compile",
    label: "torch.compile",
    section: "Performance",
    shortHint: "JIT-compile the model for faster training after a slow warmup.",
    description:
      "Applies PyTorch 2.x torch.compile to the training model. First epoch is significantly slower due to compilation; subsequent steps can be faster. Experimental — disable if you hit compatibility issues.",
    defaultValue: "false",
    recommendedValue: "false",
    valueOptions: [
      { value: "true", description: "JIT-compile model; slow first epoch, potentially faster after." },
      { value: "false", description: "Standard eager execution; more compatible." },
    ],
  },
  {
    key: "num_dataloader_workers",
    label: "DataLoader Workers",
    section: "Performance",
    shortHint: "Background threads for data loading; 0 uses the main thread.",
    description:
      "Number of worker processes for the PyTorch DataLoader. Values > 0 can overlap data loading with GPU compute. On Windows, keep at 0 unless you've verified multiprocessing stability.",
    defaultValue: "0",
    constraints: "≥ 0",
    recommendedValue: "0",
    rangeGuidance: [
      { range: "0", description: "Main-thread loading; safe default, especially on Windows." },
      { range: "2–4", description: "Background loading on Linux; overlaps I/O with GPU compute." },
    ],
  },
  {
    key: "dataloader_pin_memory",
    label: "Pin Memory",
    section: "Performance",
    shortHint: "Pin CPU memory for faster GPU transfers; requires workers > 0.",
    description:
      "Pins DataLoader memory in page-locked RAM for faster async CPU→GPU transfers. Only effective when num_dataloader_workers > 0.",
    defaultValue: "true",
    recommendedValue: "true",
    valueOptions: [
      { value: "true", description: "Faster CPU→GPU transfers when workers > 0." },
      { value: "false", description: "Standard pageable memory." },
    ],
  },
];
