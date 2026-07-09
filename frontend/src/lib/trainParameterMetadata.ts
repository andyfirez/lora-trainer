import type { ParameterMeta } from "./parameterUtils";
import { buildParameterLookup } from "./parameterUtils";

export const TRAIN_PARAMETER_METADATA: ParameterMeta[] = [
  // Model
  {
    key: "base_model_name",
    label: "Base Model",
    section: "Model",
    shortHint: "HuggingFace model ID or local path to the SDXL checkpoint used as the training base.",
    description:
      "Specifies the SDXL base model to fine-tune. Can be a HuggingFace repo ID (e.g. stabilityai/stable-diffusion-xl-base-1.0) or a local folder containing model weights. The model architecture must match SDXL; using a different base changes style, composition, and what the LoRA can learn.",
    defaultValue: "stabilityai/stable-diffusion-xl-base-1.0",
    showInlineHint: false,
    recommendedValue: "stabilityai/stable-diffusion-xl-base-1.0",
  },
  {
    key: "output_dir",
    label: "Output Folder",
    section: "Model",
    shortHint: "Directory where checkpoints, logs, and the final LoRA file are written.",
    description:
      "Root folder for all training artifacts: intermediate checkpoints, TensorBoard logs, sample images, and the exported LoRA. Use a dedicated path with enough disk space — checkpoints and cached latents can consume several GB per run.",
    defaultValue: "output",
    showInlineHint: false,
    recommendedValue: "output",
  },
  {
    key: "lora_name",
    label: "LoRA Name",
    section: "Model",
    shortHint: "Base filename for the exported LoRA; a version suffix (_vN) is appended automatically.",
    description:
      "Human-readable name for the LoRA output file. At training start the app appends a version suffix (_v1, _v2, …) to avoid overwriting previous runs. This name appears in exported .safetensors filenames and job listings.",
    defaultValue: "lora",
    showInlineHint: false,
    recommendedValue: "lora",
  },
  {
    key: "output_format",
    label: "Output Format",
    section: "Model",
    shortHint: "File format for the exported adapter weights.",
    description:
      "Controls the serialization format of the trained LoRA. safetensors is recommended for Kohya/ComfyUI/A1111 compatibility and safe loading. pt (PyTorch) is mainly useful for debugging or custom tooling.",
    defaultValue: "safetensors",
    constraints: "safetensors | pt",
    showInlineHint: false,
    recommendedValue: "safetensors",
    valueOptions: [
      { value: "safetensors", description: "Recommended — safe tensor format compatible with Kohya, ComfyUI, and A1111." },
      { value: "pt", description: "PyTorch pickle format; mainly for debugging or custom tooling." },
    ],
  },

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

  // Training
  {
    key: "epochs",
    label: "Epochs",
    section: "Training",
    shortHint: "Number of full passes over the training dataset.",
    description:
      "Total training epochs. More epochs improve learning on larger datasets but risk overfitting on small sets. Combine with repeats per concept and batch size to estimate total steps. Monitor loss and sample images to decide when to stop early.",
    defaultValue: "30",
    constraints: "≥ 1",
    showInlineHint: false,
    recommendedValue: "30",
    rangeGuidance: [
      { range: "10–20", description: "Small datasets (10–30 images); watch for overfitting." },
      { range: "30–50", description: "Medium datasets; common range for character LoRAs." },
      { range: "50+", description: "Large datasets; combine with lower learning rate and monitor samples." },
    ],
  },
  {
    key: "batch_size",
    label: "Batch Size",
    section: "Training",
    shortHint: "Images processed per optimizer step. Higher values need more VRAM.",
    description:
      "Number of training samples per forward/backward pass. SDXL LoRA training typically uses 1–4 depending on GPU VRAM. Effective batch size = batch_size × gradient_accumulation_steps.",
    defaultValue: "1",
    constraints: "≥ 1",
    showInlineHint: false,
    recommendedValue: "1",
    rangeGuidance: [
      { range: "1", description: "Minimum VRAM; standard for 12–16 GB consumer GPUs." },
      { range: "2–4", description: "Faster training on 24 GB+ GPUs; smoother gradient estimates." },
    ],
  },
  {
    key: "gradient_accumulation_steps",
    label: "Gradient Accumulation Steps",
    section: "Training",
    shortHint: "Accumulate gradients over N steps before updating weights — simulates larger batch size.",
    description:
      "Runs N forward/backward passes before each optimizer step, averaging gradients. Lets you simulate a larger batch size without proportional VRAM increase. Useful when batch_size must stay at 1 due to memory limits.",
    defaultValue: "1",
    constraints: "≥ 1",
    recommendedValue: "1",
    rangeGuidance: [
      { range: "1", description: "No accumulation; each step updates weights immediately." },
      { range: "2–8", description: "Simulates larger batch without extra VRAM; useful when batch_size must stay at 1." },
    ],
  },
  {
    key: "learning_rate",
    label: "Learning Rate",
    section: "Training",
    shortHint: "Step size for weight updates. Too high causes instability; too low trains slowly.",
    description:
      "Base learning rate for the optimizer. Typical SDXL LoRA range: 1e-5 to 1e-4 for AdamW, higher for Prodigy. The optimizer preset may override this when switching optimizer type. Pair with LR scheduler and warmup for stable training.",
    defaultValue: "5e-5",
    constraints: "> 0",
    recommendedValue: "5e-5",
    rangeGuidance: [
      { range: "1e-5", description: "Conservative; slower convergence, safer for small datasets." },
      { range: "5e-5", description: "Common AdamW default for SDXL LoRA training." },
      { range: "1e-4", description: "Aggressive; faster training but risk of instability or overfitting." },
    ],
  },
  {
    key: "lr_scheduler",
    label: "LR Scheduler",
    section: "Training",
    shortHint: "How learning rate changes over training steps.",
    description:
      "Schedule controlling learning rate decay or warmup. constant is simplest; cosine and cosine_with_restarts are popular for LoRA. constant_with_warmup and polynomial offer fine-grained control for longer runs.",
    defaultValue: "constant",
    constraints: "constant | constant_with_warmup | linear | cosine | cosine_with_restarts | polynomial",
    recommendedValue: "constant",
    valueOptions: [
      { value: "constant", description: "Fixed learning rate throughout training; simplest and reliable for short LoRA runs." },
      { value: "constant_with_warmup", description: "Ramps LR up then holds constant; good when using warmup steps." },
      { value: "linear", description: "Linear decay to zero; useful for fixed-length training schedules." },
      { value: "cosine", description: "Smooth cosine decay; popular for LoRA and often improves late-stage quality." },
      { value: "cosine_with_restarts", description: "Cosine decay with periodic restarts; helps escape plateaus on long runs." },
      { value: "polynomial", description: "Polynomial decay curve; fine-grained control for custom schedules." },
    ],
  },
  {
    key: "lr_warmup_steps",
    label: "LR Warmup Steps",
    section: "Training",
    shortHint: "Gradually ramp learning rate from zero over this many steps.",
    description:
      "Number of steps to linearly warm up the learning rate from a small value to the target LR. Helps stabilize early training, especially with large learning rates or Prodigy. Set to 0 to disable warmup.",
    defaultValue: "0",
    constraints: "≥ 0",
    recommendedValue: "0",
    rangeGuidance: [
      { range: "0", description: "No warmup; fine for standard AdamW with moderate learning rates." },
      { range: "50–200", description: "Gentle ramp for large LRs or Prodigy optimizer." },
      { range: "5–10% of total steps", description: "Proportional warmup for long training runs." },
    ],
  },
  {
    key: "min_snr_gamma",
    label: "Min SNR Gamma",
    section: "Training",
    shortHint: "Loss reweighting for noisy timesteps; 5 is a common SDXL value, 0 disables.",
    description:
      "Minimum SNR gamma for loss reweighting (Min-SNR weighting strategy). Reduces loss contribution from very noisy timesteps, often improving convergence. Kohya default is 5. Set to 0 to use standard uniform weighting.",
    defaultValue: "5.0",
    constraints: "≥ 0",
    recommendedValue: "5.0",
    rangeGuidance: [
      { range: "0", description: "Disables Min-SNR weighting; uniform loss across timesteps." },
      { range: "5", description: "Kohya SDXL default; often improves convergence." },
      { range: "10+", description: "Stronger reweighting; experiment if loss plateaus early." },
    ],
  },
  {
    key: "noise_offset",
    label: "Noise Offset",
    section: "Training",
    shortHint: "Adds slight noise bias to improve very dark/bright image learning.",
    description:
      "Small constant offset added to training noise. Helps the model learn extreme brightness values (pure black/white regions). Kohya SDXL default is ~0.0357. Set to 0 to disable.",
    defaultValue: "0.0357",
    constraints: "≥ 0",
    recommendedValue: "0.0357",
    rangeGuidance: [
      { range: "0", description: "Disabled; standard uniform noise distribution." },
      { range: "0.0357", description: "Kohya SDXL default; helps learn extreme brightness values." },
      { range: "0.05–0.1", description: "Stronger offset for datasets with very dark or bright images." },
    ],
  },

  // Optimizer
  {
    key: "optimizer.type",
    label: "Optimizer Type",
    section: "Optimizer",
    shortHint: "Optimization algorithm; adamw_8bit is the default balance of speed and memory.",
    description:
      "Selects the optimizer implementation. adamw_8bit (bitsandbytes) is the default — good VRAM efficiency. adamw is full-precision AdamW. adafactor is memory-efficient for large models. prodigy adapts learning rate automatically but needs tuning.",
    defaultValue: "adamw_8bit",
    constraints: "adamw | adamw_8bit | adafactor | prodigy",
    recommendedValue: "adamw_8bit",
    valueOptions: [
      { value: "adamw", description: "Full-precision AdamW; stable but uses more VRAM for optimizer states." },
      { value: "adamw_8bit", description: "8-bit quantized AdamW via bitsandbytes; default balance of speed and memory." },
      { value: "adafactor", description: "Memory-efficient factored optimizer; good for very large models." },
      { value: "prodigy", description: "Adaptive LR optimizer; can eliminate manual LR tuning but needs careful monitoring." },
    ],
  },
  {
    key: "optimizer.weight_decay",
    label: "Weight Decay",
    section: "Optimizer",
    shortHint: "L2 regularization strength; reduces overfitting.",
    description:
      "Coefficient for weight decay (L2 penalty) applied by Adam-family and Prodigy optimizers. Typical range 0.01–0.1. Higher values increase regularization; 0 disables decay.",
    defaultValue: "0.01",
    constraints: "≥ 0",
    recommendedValue: "0.01",
    rangeGuidance: [
      { range: "0", description: "No L2 regularization." },
      { range: "0.01", description: "Light regularization; common default." },
      { range: "0.05–0.1", description: "Stronger regularization when overfitting on small datasets." },
    ],
  },
  {
    key: "optimizer.beta1",
    label: "Beta 1",
    section: "Optimizer",
    shortHint: "Adam first moment decay rate.",
    description:
      "Exponential decay rate for the first moment estimate in Adam/Prodigy optimizers. Standard default is 0.9. Rarely needs changing for LoRA training.",
    defaultValue: "0.9",
    constraints: "(0, 1)",
    recommendedValue: "0.9",
    rangeGuidance: [
      { range: "0.9", description: "Standard Adam default; rarely needs changing." },
      { range: "0.95", description: "Slower first-moment decay; smoother updates for noisy gradients." },
    ],
  },
  {
    key: "optimizer.beta2",
    label: "Beta 2",
    section: "Optimizer",
    shortHint: "Adam second moment decay rate.",
    description:
      "Exponential decay rate for the second moment estimate in Adam/Prodigy optimizers. Standard default is 0.999.",
    defaultValue: "0.999",
    constraints: "(0, 1)",
    recommendedValue: "0.999",
    rangeGuidance: [
      { range: "0.999", description: "Standard Adam default." },
      { range: "0.99", description: "Faster second-moment adaptation; can help with sparse gradients." },
    ],
  },
  {
    key: "optimizer.relative_step",
    label: "Relative Step",
    section: "Optimizer",
    shortHint: "Adafactor: scale step size relative to parameter scale.",
    description:
      "Adafactor-only flag. When true, Adafactor computes relative step sizes based on parameter magnitudes, often eliminating the need for a manual learning rate.",
    defaultValue: "false",
    recommendedValue: "false",
    valueOptions: [
      { value: "true", description: "Adafactor computes relative step sizes from parameter scale." },
      { value: "false", description: "Use fixed learning rate with Adafactor." },
    ],
  },
  {
    key: "optimizer.scale_parameter",
    label: "Scale Parameter",
    section: "Optimizer",
    shortHint: "Adafactor: apply per-parameter scaling.",
    description:
      "Adafactor-only flag. Enables factored second-moment scaling per parameter group. Usually left false unless using relative_step mode.",
    defaultValue: "false",
    recommendedValue: "false",
    valueOptions: [
      { value: "true", description: "Enable per-parameter factored scaling in Adafactor." },
      { value: "false", description: "Disable factored scaling." },
    ],
  },
  {
    key: "optimizer.warmup_init",
    label: "Warmup Init",
    section: "Optimizer",
    shortHint: "Adafactor: initialize with warmup schedule.",
    description:
      "Adafactor-only flag. When true, uses a warmup initialization scheme for the learning rate schedule inside Adafactor.",
    defaultValue: "false",
    recommendedValue: "false",
    valueOptions: [
      { value: "true", description: "Use warmup initialization in Adafactor schedule." },
      { value: "false", description: "No Adafactor warmup init." },
    ],
  },
  {
    key: "optimizer.decouple",
    label: "Decouple",
    section: "Optimizer",
    shortHint: "Prodigy: decouple weight decay from gradient update.",
    description:
      "Prodigy-only flag. Decoupled weight decay (AdamW-style) vs. L2 regularization coupled with gradients. True is recommended for Prodigy.",
    defaultValue: "true",
    recommendedValue: "true",
    valueOptions: [
      { value: "true", description: "AdamW-style decoupled weight decay for Prodigy." },
      { value: "false", description: "Coupled L2 regularization with gradients." },
    ],
  },
  {
    key: "optimizer.use_bias_correction",
    label: "Use Bias Correction",
    section: "Optimizer",
    shortHint: "Prodigy: apply bias correction to moment estimates.",
    description:
      "Prodigy-only flag. Enables bias correction for adaptive moment estimates, similar to standard Adam.",
    defaultValue: "true",
    recommendedValue: "true",
    valueOptions: [
      { value: "true", description: "Apply bias correction to Prodigy moment estimates." },
      { value: "false", description: "Skip bias correction." },
    ],
  },
  {
    key: "optimizer.safeguard_warmup",
    label: "Safeguard Warmup",
    section: "Optimizer",
    shortHint: "Prodigy: protect early steps from unstable adaptive LR.",
    description:
      "Prodigy-only flag. Adds safeguards during warmup to prevent excessively large adaptive learning rates at the start of training.",
    defaultValue: "true",
    recommendedValue: "true",
    valueOptions: [
      { value: "true", description: "Protect early Prodigy steps from unstable adaptive LR." },
      { value: "false", description: "No warmup safeguards." },
    ],
  },
  {
    key: "optimizer.d0",
    label: "d0",
    section: "Optimizer",
    shortHint: "Prodigy: initial estimate for D parameter.",
    description:
      "Prodigy-only hyperparameter controlling the initial value of the D estimate used for adaptive learning rate scaling. Default 1e-5 works for most LoRA runs.",
    defaultValue: "1e-5",
    constraints: "> 0",
    recommendedValue: "1e-5",
    rangeGuidance: [
      { range: "1e-6", description: "Smaller initial D; more conservative adaptive LR." },
      { range: "1e-5", description: "Prodigy default; works for most LoRA runs." },
      { range: "1e-4", description: "Larger initial D; faster early adaptation, monitor for instability." },
    ],
  },
  {
    key: "optimizer.d_coef",
    label: "d Coef",
    section: "Optimizer",
    shortHint: "Prodigy: coefficient for D estimate updates.",
    description:
      "Prodigy-only scaling coefficient for the D parameter update rule. Default 1.0; increase cautiously if training is too slow.",
    defaultValue: "1.0",
    constraints: "> 0",
    recommendedValue: "1.0",
    rangeGuidance: [
      { range: "0.5", description: "Slower D updates; more stable but slower convergence." },
      { range: "1.0", description: "Default coefficient." },
      { range: "2.0", description: "Faster D adaptation; use cautiously." },
    ],
  },

  // Data
  {
    key: "resolution",
    label: "Resolution",
    section: "Data",
    shortHint: "Training image resolution in pixels. SDXL default is 1024.",
    description:
      "Target resolution for training images. SDXL is trained at 1024×1024; datasets should be preprocessed to match. With bucketing enabled, images are grouped into aspect-ratio buckets near this resolution.",
    defaultValue: "1024",
    constraints: "64–2048",
    showInlineHint: false,
    recommendedValue: "1024",
    rangeGuidance: [
      { range: "512–768", description: "Lower resolution; faster training, less detail for SDXL." },
      { range: "1024", description: "SDXL native resolution; recommended default." },
      { range: "1280–1536", description: "Higher detail; significantly more VRAM per image." },
    ],
  },
  {
    key: "enable_bucket",
    label: "Enable Bucketing",
    section: "Data",
    shortHint: "Group images by aspect ratio instead of forcing square crops.",
    description:
      "When enabled, images are assigned to resolution buckets preserving aspect ratio, reducing distortion from squashing non-square images. Dataset preprocessing must match this setting.",
    defaultValue: "false",
    showInlineHint: false,
    recommendedValue: "false",
    valueOptions: [
      { value: "true", description: "Preserve aspect ratios via resolution buckets." },
      { value: "false", description: "Square crop/resize all images to resolution." },
    ],
  },
  {
    key: "bucket_reso_steps",
    label: "Bucket Resolution Steps",
    section: "Data",
    shortHint: "Step size between bucket resolutions.",
    description:
      "Granularity of bucket sizes when aspect-ratio bucketing is enabled. Smaller steps give finer aspect-ratio matching but more buckets and potential batch inefficiency.",
    defaultValue: "64",
    constraints: "8–512",
    yamlOnly: true,
    recommendedValue: "64",
    rangeGuidance: [
      { range: "32", description: "Fine-grained buckets; better aspect-ratio matching, more buckets." },
      { range: "64", description: "Standard step size; good balance." },
      { range: "128", description: "Coarser buckets; fewer buckets, less precise aspect matching." },
    ],
  },
  {
    key: "min_bucket_reso",
    label: "Min Bucket Resolution",
    section: "Data",
    shortHint: "Smallest bucket edge length.",
    description:
      "Minimum bucket resolution edge when bucketing is enabled. Images smaller than this may be upscaled or filtered depending on bucket_no_upscale.",
    defaultValue: "512",
    constraints: "64–2048",
    yamlOnly: true,
    recommendedValue: "512",
    rangeGuidance: [
      { range: "256–512", description: "Filters very small images; reduces upscaling artifacts." },
      { range: "768", description: "Higher minimum; excludes low-res training images." },
    ],
  },
  {
    key: "max_bucket_reso",
    label: "Max Bucket Resolution",
    section: "Data",
    shortHint: "Largest bucket edge length.",
    description:
      "Maximum bucket resolution edge when bucketing is enabled. Caps VRAM usage for very large aspect-ratio images.",
    defaultValue: "2048",
    constraints: "64–2048",
    yamlOnly: true,
    recommendedValue: "2048",
    rangeGuidance: [
      { range: "1024", description: "Caps VRAM for large aspect-ratio images." },
      { range: "1536–2048", description: "Allows high-res buckets; needs more VRAM." },
    ],
  },
  {
    key: "bucket_no_upscale",
    label: "Bucket No Upscale",
    section: "Data",
    shortHint: "Prevent upscaling images to fit larger buckets.",
    description:
      "When true, images are not upscaled to fit a larger bucket — they stay at native resolution within bucket constraints. Reduces artifacts from upscaling small source images.",
    defaultValue: "true",
    yamlOnly: true,
    recommendedValue: "true",
    valueOptions: [
      { value: "true", description: "Keep native resolution; no upscaling to fit buckets." },
      { value: "false", description: "Allow upscaling small images into larger buckets." },
    ],
  },
  {
    key: "concepts.dataset_id",
    label: "Dataset",
    section: "Data",
    shortHint: "Dataset providing images and captions for this concept.",
    description:
      "References a preprocessed dataset by ID. The dataset must match training resolution and bucketing settings. Images and captions are loaded from the dataset's prepared directory at training time.",
    recommendedValue: "your-dataset-id",
  },
  {
    key: "concepts.trigger_words",
    label: "Trigger Words",
    section: "Data",
    shortHint: "Tokens prepended to captions to activate the LoRA at inference.",
    description:
      "Comma-separated trigger words inserted into training captions and sample prompts. These tokens become associated with the learned concept — use unique, rare tokens (e.g. ohwx, sks) to avoid conflicts with base model vocabulary.",
    recommendedValue: "ohwx, unique token",
  },
  {
    key: "concepts.caption_extension",
    label: "Caption Extension",
    section: "Data",
    shortHint: "File extension for caption sidecar files.",
    description:
      "Extension of caption files alongside images (e.g. .txt for image.jpg → image.txt). Must match how captions were exported from your tagging workflow.",
    defaultValue: ".txt",
    recommendedValue: ".txt",
  },
  {
    key: "concepts.repeats",
    label: "Repeats",
    section: "Data",
    shortHint: "How many times each image in this concept is seen per epoch.",
    description:
      "Multiplies each image's contribution per epoch. Increase repeats for small datasets to give the model more exposure without adding epochs. Schema default is 3; the form defaults new concepts to 1.",
    defaultValue: "3",
    constraints: "≥ 1",
    recommendedValue: "3",
    rangeGuidance: [
      { range: "1", description: "Each image seen once per epoch; form default for new concepts." },
      { range: "3–5", description: "Common for small datasets (10–20 images)." },
      { range: "10+", description: "Heavy repetition for very small sets; watch for overfitting." },
    ],
  },
  {
    key: "concepts.caption_suffix",
    label: "Caption Suffix",
    section: "Data",
    shortHint: "Text appended to every caption in this concept.",
    description:
      "Optional suffix added to all captions for this concept after loading. Useful for adding consistent style tags or quality tokens across a dataset.",
    defaultValue: '""',
    yamlOnly: true,
    recommendedValue: '""',
  },
  {
    key: "concepts.image_dir",
    label: "Image Directory",
    section: "Data",
    shortHint: "Deprecated — resolved automatically from dataset_id at runtime.",
    description:
      "Legacy field for the raw image directory path. Modern configs use concepts.dataset_id instead; the trainer resolves image_dir from the dataset record when the job starts. Do not set manually in new configs.",
    yamlOnly: true,
    deprecated: true,
    recommendedValue: "do not set (use concepts.dataset_id)",
  },
  {
    key: "concepts.prepared_dir",
    label: "Prepared Directory",
    section: "Data",
    shortHint: "Deprecated — resolved automatically from dataset_id at runtime.",
    description:
      "Legacy field for the preprocessed dataset directory. Modern configs use concepts.dataset_id instead; the trainer resolves prepared_dir from the dataset record when the job starts. Do not set manually in new configs.",
    yamlOnly: true,
    deprecated: true,
    recommendedValue: "do not set (use concepts.dataset_id)",
  },

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
    yamlOnly: true,
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

  // Checkpointing
  {
    key: "checkpointing_enabled",
    label: "Enable Checkpoints",
    section: "Checkpointing",
    shortHint: "Save intermediate LoRA weights during training.",
    description:
      "When enabled, saves LoRA checkpoints at regular epoch intervals. Required for mid-training sampling and resume. Disabling also disables sampling during training.",
    defaultValue: "true",
    showInlineHint: false,
    recommendedValue: "true",
    valueOptions: [
      { value: "true", description: "Save intermediate weights; required for sampling and resume." },
      { value: "false", description: "Only save final LoRA; disables mid-training sampling." },
    ],
  },
  {
    key: "save_every_n_epochs",
    label: "Save Every N Epochs",
    section: "Checkpointing",
    shortHint: "Checkpoint frequency in epochs.",
    description:
      "Interval between checkpoint saves. Set to 1 to save every epoch. Larger values reduce disk usage but give fewer restore points and sample images.",
    defaultValue: "1",
    constraints: "≥ 1",
    recommendedValue: "1",
    rangeGuidance: [
      { range: "1", description: "Save every epoch; most restore points and sample images." },
      { range: "5–10", description: "Less disk usage; fewer checkpoints to compare." },
    ],
  },
  {
    key: "resume_from_checkpoint",
    label: "Resume From Checkpoint",
    section: "Checkpointing",
    shortHint: "Path to a checkpoint folder to continue a previous run.",
    description:
      "Filesystem path to an existing checkpoint directory. Training resumes optimizer state, epoch counter, and LoRA weights from this checkpoint. Set via YAML for resume workflows.",
    yamlOnly: true,
    recommendedValue: "path/to/checkpoint",
  },

  // Sampling
  {
    key: "sampling_enabled",
    label: "Sampling Enabled",
    section: "Sampling",
    shortHint: "Generate preview images at checkpoints using a linked sampling config.",
    description:
      "Runs image generation after checkpoint saves using prompts from the linked sampling config. Helps monitor training quality without manual inference. Requires checkpointing enabled.",
    defaultValue: "false",
    showInlineHint: false,
    recommendedValue: "false",
    valueOptions: [
      { value: "true", description: "Generate preview images at checkpoints." },
      { value: "false", description: "No mid-training previews." },
    ],
  },
  {
    key: "sampling_config_id",
    label: "Sampling Config",
    section: "Sampling",
    shortHint: "Reference to a saved sampling config with prompts and sampler settings.",
    description:
      "ID of a sampling configuration stored in the app database. Its prompts, steps, CFG, and scheduler settings are used for mid-training preview generation.",
    recommendedValue: "linked sampling config",
  },
  {
    key: "sample_every_n_epochs",
    label: "Sample Every N Epochs",
    section: "Sampling",
    shortHint: "Override checkpoint interval specifically for sampling.",
    description:
      "If set, controls sampling frequency independently of save_every_n_epochs. When null, sampling runs on every checkpoint save.",
    yamlOnly: true,
    recommendedValue: "null (uses save_every_n_epochs)",
  },
  {
    key: "sample_before_training",
    label: "Sample Before Training",
    section: "Sampling",
    shortHint: "Generate baseline images before any training steps.",
    description:
      "When true, runs sampling once before training begins to capture base-model output for comparison with trained checkpoints.",
    defaultValue: "false",
    yamlOnly: true,
    recommendedValue: "false",
    valueOptions: [
      { value: "true", description: "Capture base-model baseline before training." },
      { value: "false", description: "Skip pre-training preview." },
    ],
  },
  {
    key: "sample_prompts",
    label: "Sample Prompts",
    section: "Sampling",
    shortHint: "Prompts used for mid-training preview images (resolved from sampling config).",
    description:
      "List of text prompts for checkpoint preview generation. In the web UI these come from the linked sampling config via resolve_sampling(); inline YAML values are merged at job start. Trigger words from concepts are appended automatically.",
    defaultValue: "[]",
    yamlOnly: true,
    recommendedValue: "[]",
  },
  {
    key: "sample_negative_prompt",
    label: "Sample Negative Prompt",
    section: "Sampling",
    shortHint: "Negative prompt for mid-training preview generation.",
    description:
      "Negative prompt text applied during checkpoint sampling. Typically resolved from the linked sampling config rather than set inline in training YAML.",
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
      "Inference steps for mid-training sample generation. Higher values improve preview quality at the cost of slower sampling after each checkpoint. Resolved from the linked sampling config in normal workflows.",
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
      "CFG scale controlling prompt adherence during checkpoint sampling. SDXL commonly uses 5–8. Resolved from the linked sampling config in normal workflows.",
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
      "Width in pixels for generated preview images. When null, the trainer uses the training resolution. Resolved from the linked sampling config in normal workflows.",
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
      "Height in pixels for generated preview images. When null, the trainer uses the training resolution. Resolved from the linked sampling config in normal workflows.",
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
      "Sampler algorithm for mid-training previews. euler is the default and matches most SDXL workflows. Resolved from the linked sampling config in normal workflows.",
    defaultValue: "euler",
    constraints: "euler | euler_a | ddim | dpm++",
    yamlOnly: true,
    recommendedValue: "euler",
    valueOptions: [
      { value: "euler", description: "Euler sampler; fast and matches most SDXL workflows." },
      { value: "euler_a", description: "Euler ancestral; adds stochasticity for more varied previews." },
      { value: "ddim", description: "DDIM deterministic sampler; reproducible previews." },
      { value: "dpm++", description: "DPM++ solver; higher quality previews at the cost of slower sampling." },
    ],
  },
  {
    key: "sample_vae_tiling",
    label: "Sample VAE Tiling",
    section: "Sampling",
    shortHint: "Tile VAE decode to reduce VRAM during preview generation.",
    description:
      "Enables tiled VAE decoding during checkpoint sampling, trading a small speed penalty for lower peak VRAM usage when generating large preview images.",
    defaultValue: "true",
    yamlOnly: true,
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
      "Offloads the UNet from GPU memory before VAE decoding during checkpoint sampling. Helps avoid OOM on consumer GPUs when generating previews at full resolution.",
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
    shortHint: "Deprecated — use sampling_config_id with sampling_enabled instead.",
    description:
      "Legacy field for linking a sampling config after training completes. Rejected by config validation in favor of sampling_enabled + sampling_config_id for mid-training previews. Do not use in new configs.",
    yamlOnly: true,
    deprecated: true,
    recommendedValue: "do not use (use sampling_config_id + sampling_enabled)",
  },
  {
    key: "sample_after_training",
    label: "Sample After Training",
    section: "Sampling",
    shortHint: "Deprecated — removed; use sampling_enabled for preview generation.",
    description:
      "Deprecated boolean flag for post-training sampling. Config validation rejects this key. Use sampling_enabled with a linked sampling config for preview images during training.",
    yamlOnly: true,
    deprecated: true,
    recommendedValue: "do not use (use sampling_enabled)",
  },

  // Logging
  {
    key: "logging.use_ui_logger",
    label: "UI Logger",
    section: "Logging",
    shortHint: "Stream training metrics to the web UI.",
    description:
      "Enables the built-in UI logger that pushes loss and progress metrics to the job detail page in real time.",
    defaultValue: "true",
    yamlOnly: true,
    recommendedValue: "true",
    valueOptions: [
      { value: "true", description: "Stream metrics to the web UI job page." },
      { value: "false", description: "File-only logging." },
    ],
  },
  {
    key: "logging.log_every",
    label: "Log Every",
    section: "Logging",
    shortHint: "Log metrics every N steps.",
    description:
      "Step interval for writing training metrics to logs and the UI. Lower values give smoother charts but slightly more I/O overhead.",
    defaultValue: "1",
    constraints: "≥ 1",
    yamlOnly: true,
    recommendedValue: "1",
    rangeGuidance: [
      { range: "1", description: "Log every step; smoothest UI charts." },
      { range: "10–50", description: "Reduced I/O for long runs with many steps per epoch." },
    ],
  },
  {
    key: "logging.log_dir",
    label: "Log Directory",
    section: "Logging",
    shortHint: "Custom directory for TensorBoard and log files.",
    description:
      "Optional override for log output directory. When null, logs are written under output_dir.",
    yamlOnly: true,
    recommendedValue: "null (uses output_dir)",
  },
];

export const TRAIN_PARAMETER_LOOKUP = buildParameterLookup(TRAIN_PARAMETER_METADATA);

export function getTrainParameterMeta(key: string): ParameterMeta | undefined {
  return TRAIN_PARAMETER_LOOKUP.get(key);
}

export function trainHint(key: string): { hint?: string; hintAnchor?: string } {
  const meta = getTrainParameterMeta(key);
  if (!meta || meta.showInlineHint === false) return {};
  return { hint: meta.shortHint, hintAnchor: meta.key };
}
