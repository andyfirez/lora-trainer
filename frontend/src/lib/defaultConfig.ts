export const TrainConfig = {
  DEFAULT_YAML: `# SDXL LoRA Training Configuration
base_model_name: stabilityai/stable-diffusion-xl-base-1.0
output_dir: output
lora_name: lora
output_format: safetensors

# LoRA settings
lora_rank: 16
lora_alpha: 16.0
lora_dropout: 0.0

# Training targets
unet:
  train: true
  weight_dtype: float16
text_encoder_1:
  train: false
  weight_dtype: float16
text_encoder_2:
  train: false
  weight_dtype: float16

# Training hyperparameters
epochs: 10
batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 0.0001
lr_scheduler: constant
lr_warmup_steps: 0
optimizer: adamw

# Data
resolution: 1024
concepts:
  - image_dir: /path/to/images
    caption_extension: .txt
    repeats: 1

# Optimization
gradient_checkpointing: true
mixed_precision: float16

# Checkpointing & Sampling
save_every_n_epochs: 1
sample_every_n_epochs: null
sample_before_training: false
sample_after_training: false
sample_prompts: []
sample_negative_prompt: ''
sample_steps: 30
sample_cfg_scale: 7.5
sample_scheduler: euler
use_reforge_sampler: false
sample_sampler: euler_a
sample_scheduler_mode: normal

# Logging (loss graph + optional TensorBoard)
logging:
  use_ui_logger: true
  log_every: 1
  # log_dir: output/.tensorboard
`,
};

export const SamplingConfig = {
  DEFAULT_YAML: `# SDXL LoRA Sampling Configuration
base_model_name: stabilityai/stable-diffusion-xl-base-1.0
output_dir: output
lora_name: lora
sample_prompts: []
sample_negative_prompt: ''
sample_steps: 30
sample_cfg_scale: 7.5
# sample_width: 1024
# sample_height: 1024
sample_scheduler: euler
use_reforge_sampler: false
sample_sampler: euler_a
sample_scheduler_mode: normal
attention_mechanism: sdpa
mixed_precision: float16
vae_dtype: auto
tf32: true
lora_paths: []
`,
};
