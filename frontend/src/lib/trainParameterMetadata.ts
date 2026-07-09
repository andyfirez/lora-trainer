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
  },
  {
    key: "output_dir",
    label: "Output Folder",
    section: "Model",
    shortHint: "Directory where checkpoints, logs, and the final LoRA file are written.",
    description:
      "Root folder for all training artifacts: intermediate checkpoints, TensorBoard logs, sample images, and the exported LoRA. Use a dedicated path with enough disk space — checkpoints and cached latents can consume several GB per run.",
    defaultValue: "output",
  },
  {
    key: "lora_name",
    label: "LoRA Name",
    section: "Model",
    shortHint: "Base filename for the exported LoRA; a version suffix (_vN) is appended automatically.",
    description:
      "Human-readable name for the LoRA output file. At training start the app appends a version suffix (_v1, _v2, …) to avoid overwriting previous runs. This name appears in exported .safetensors filenames and job listings.",
    defaultValue: "lora",
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
  },
  {
    key: "text_encoder_1.train",
    label: "Text Encoder 1 — Train",
    section: "Training Targets",
    shortHint: "Train LoRA on CLIP-L text encoder. Increases VRAM and training time.",
    description:
      "Enables LoRA fine-tuning on the first SDXL text encoder (CLIP ViT-L). Useful when trigger words or captions need stronger semantic binding. Incompatible with caching text encoder outputs while training TEs.",
    defaultValue: "false",
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
  },
  {
    key: "text_encoder_2.train",
    label: "Text Encoder 2 — Train",
    section: "Training Targets",
    shortHint: "Train LoRA on OpenCLIP-G text encoder. Rarely needed for most LoRAs.",
    description:
      "Enables LoRA fine-tuning on SDXL's second text encoder (OpenCLIP-G). Typically left disabled unless you need fine-grained caption semantics. Significantly increases memory when combined with UNet training.",
    defaultValue: "false",
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
  },
  {
    key: "optimizer.relative_step",
    label: "Relative Step",
    section: "Optimizer",
    shortHint: "Adafactor: scale step size relative to parameter scale.",
    description:
      "Adafactor-only flag. When true, Adafactor computes relative step sizes based on parameter magnitudes, often eliminating the need for a manual learning rate.",
    defaultValue: "false",
  },
  {
    key: "optimizer.scale_parameter",
    label: "Scale Parameter",
    section: "Optimizer",
    shortHint: "Adafactor: apply per-parameter scaling.",
    description:
      "Adafactor-only flag. Enables factored second-moment scaling per parameter group. Usually left false unless using relative_step mode.",
    defaultValue: "false",
  },
  {
    key: "optimizer.warmup_init",
    label: "Warmup Init",
    section: "Optimizer",
    shortHint: "Adafactor: initialize with warmup schedule.",
    description:
      "Adafactor-only flag. When true, uses a warmup initialization scheme for the learning rate schedule inside Adafactor.",
    defaultValue: "false",
  },
  {
    key: "optimizer.decouple",
    label: "Decouple",
    section: "Optimizer",
    shortHint: "Prodigy: decouple weight decay from gradient update.",
    description:
      "Prodigy-only flag. Decoupled weight decay (AdamW-style) vs. L2 regularization coupled with gradients. True is recommended for Prodigy.",
    defaultValue: "true",
  },
  {
    key: "optimizer.use_bias_correction",
    label: "Use Bias Correction",
    section: "Optimizer",
    shortHint: "Prodigy: apply bias correction to moment estimates.",
    description:
      "Prodigy-only flag. Enables bias correction for adaptive moment estimates, similar to standard Adam.",
    defaultValue: "true",
  },
  {
    key: "optimizer.safeguard_warmup",
    label: "Safeguard Warmup",
    section: "Optimizer",
    shortHint: "Prodigy: protect early steps from unstable adaptive LR.",
    description:
      "Prodigy-only flag. Adds safeguards during warmup to prevent excessively large adaptive learning rates at the start of training.",
    defaultValue: "true",
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
  },
  {
    key: "enable_bucket",
    label: "Enable Bucketing",
    section: "Data",
    shortHint: "Group images by aspect ratio instead of forcing square crops.",
    description:
      "When enabled, images are assigned to resolution buckets preserving aspect ratio, reducing distortion from squashing non-square images. Dataset preprocessing must match this setting.",
    defaultValue: "false",
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
  },
  {
    key: "concepts.dataset_id",
    label: "Dataset",
    section: "Data",
    shortHint: "Dataset providing images and captions for this concept.",
    description:
      "References a preprocessed dataset by ID. The dataset must match training resolution and bucketing settings. Images and captions are loaded from the dataset's prepared directory at training time.",
  },
  {
    key: "concepts.trigger_words",
    label: "Trigger Words",
    section: "Data",
    shortHint: "Tokens prepended to captions to activate the LoRA at inference.",
    description:
      "Comma-separated trigger words inserted into training captions and sample prompts. These tokens become associated with the learned concept — use unique, rare tokens (e.g. ohwx, sks) to avoid conflicts with base model vocabulary.",
  },
  {
    key: "concepts.caption_extension",
    label: "Caption Extension",
    section: "Data",
    shortHint: "File extension for caption sidecar files.",
    description:
      "Extension of caption files alongside images (e.g. .txt for image.jpg → image.txt). Must match how captions were exported from your tagging workflow.",
    defaultValue: ".txt",
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
  },
  {
    key: "seed",
    label: "Seed",
    section: "Optimization",
    shortHint: "Random seed for reproducibility; leave empty for random.",
    description:
      "Fixes random number generation for dataset shuffling, noise sampling, and weight initialization. Set a specific integer to reproduce a training run. Omit or leave empty for a random seed each run.",
    defaultValue: "random",
  },
  {
    key: "gradient_checkpointing",
    label: "Gradient Checkpointing",
    section: "Optimization",
    shortHint: "Trade compute for VRAM by recomputing activations during backward pass.",
    description:
      "Recomputes intermediate activations during backprop instead of storing them, significantly reducing VRAM at the cost of ~20–30% slower training. Strongly recommended for SDXL LoRA on consumer GPUs.",
    defaultValue: "true",
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
  },
  {
    key: "cache_latents_to_disk",
    label: "Cache Latents to Disk",
    section: "Performance",
    shortHint: "Persist VAE latents as .npz files; requires RAM caching enabled.",
    description:
      "Saves encoded latents to disk (.npz) so subsequent runs skip VAE encoding entirely. Requires cache_latents enabled. Useful for iterative hyperparameter tuning on the same dataset.",
    defaultValue: "false",
  },
  {
    key: "cache_text_encoder_outputs",
    label: "Cache Text Encoder Outputs (RAM)",
    section: "Performance",
    shortHint: "Pre-compute text embeddings in RAM. Incompatible with training text encoders.",
    description:
      "Caches CLIP text encoder outputs for all captions in RAM, skipping text encoding each step. Cannot be used while training text encoders (text_encoder_1.train or text_encoder_2.train is true).",
    defaultValue: "true",
  },
  {
    key: "cache_text_encoder_outputs_to_disk",
    label: "Cache Text Encoder Outputs to Disk",
    section: "Performance",
    shortHint: "Persist text embeddings to disk; requires RAM TE caching.",
    description:
      "Saves text encoder outputs to disk for reuse across runs. Requires cache_text_encoder_outputs enabled.",
    defaultValue: "false",
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
  },
  {
    key: "tf32",
    label: "TF32",
    section: "Performance",
    shortHint: "Use TensorFloat-32 on Ampere+ GPUs for faster matmuls.",
    description:
      "Enables TF32 mode for matrix multiplications on NVIDIA Ampere and newer GPUs. Provides a free speed boost with negligible quality impact for most LoRA training.",
    defaultValue: "true",
  },
  {
    key: "torch_compile",
    label: "torch.compile",
    section: "Performance",
    shortHint: "JIT-compile the model for faster training after a slow warmup.",
    description:
      "Applies PyTorch 2.x torch.compile to the training model. First epoch is significantly slower due to compilation; subsequent steps can be faster. Experimental — disable if you hit compatibility issues.",
    defaultValue: "false",
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
  },
  {
    key: "dataloader_pin_memory",
    label: "Pin Memory",
    section: "Performance",
    shortHint: "Pin CPU memory for faster GPU transfers; requires workers > 0.",
    description:
      "Pins DataLoader memory in page-locked RAM for faster async CPU→GPU transfers. Only effective when num_dataloader_workers > 0.",
    defaultValue: "true",
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
  },
  {
    key: "resume_from_checkpoint",
    label: "Resume From Checkpoint",
    section: "Checkpointing",
    shortHint: "Path to a checkpoint folder to continue a previous run.",
    description:
      "Filesystem path to an existing checkpoint directory. Training resumes optimizer state, epoch counter, and LoRA weights from this checkpoint. Set via YAML for resume workflows.",
    yamlOnly: true,
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
  },
  {
    key: "sampling_config_id",
    label: "Sampling Config",
    section: "Sampling",
    shortHint: "Reference to a saved sampling config with prompts and sampler settings.",
    description:
      "ID of a sampling configuration stored in the app database. Its prompts, steps, CFG, and scheduler settings are used for mid-training preview generation.",
  },
  {
    key: "sample_every_n_epochs",
    label: "Sample Every N Epochs",
    section: "Sampling",
    shortHint: "Override checkpoint interval specifically for sampling.",
    description:
      "If set, controls sampling frequency independently of save_every_n_epochs. When null, sampling runs on every checkpoint save.",
    yamlOnly: true,
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
  },
  {
    key: "logging.log_dir",
    label: "Log Directory",
    section: "Logging",
    shortHint: "Custom directory for TensorBoard and log files.",
    description:
      "Optional override for log output directory. When null, logs are written under output_dir.",
    yamlOnly: true,
  },
];

export const TRAIN_PARAMETER_LOOKUP = buildParameterLookup(TRAIN_PARAMETER_METADATA);

export function getTrainParameterMeta(key: string): ParameterMeta | undefined {
  return TRAIN_PARAMETER_LOOKUP.get(key);
}

export function trainHint(key: string): { hint?: string; hintAnchor?: string } {
  const meta = getTrainParameterMeta(key);
  if (!meta) return {};
  return { hint: meta.shortHint, hintAnchor: meta.key };
}
