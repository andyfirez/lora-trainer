export const TrainConfig = {
  DEFAULT_YAML: `# SDXL LoRA Training Configuration
base_model_name: stabilityai/stable-diffusion-xl-base-1.0
output_dir: output
lora_name: lora
output_format: safetensors

# LoRA settings
lora_rank: 32
lora_alpha: 32.0
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
epochs: 30
batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 0.00005
lr_scheduler: constant
lr_warmup_steps: 0
min_snr_gamma: 5.0
noise_offset: 0.0357
clip_skip: 2

optimizer:
  type: adamw_8bit
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.999
  relative_step: false
  scale_parameter: false
  warmup_init: false
  decouple: true
  use_bias_correction: true
  safeguard_warmup: true
  d0: 0.00001
  d_coef: 1.0

# Data
resolution: 1024
concepts:
  - dataset_id: 1
    trigger_words: []
    caption_extension: .txt
    repeats: 3

# Optimization
gradient_checkpointing: true
mixed_precision: float16

# Checkpointing
checkpointing_enabled: true
save_every_n_epochs: 1

# Sampling
sampling_enabled: false
# sampling_config_id: 1

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
sample_prompts: []
sample_negative_prompt: ''
sample_steps: 30
sample_cfg_scale: 7.5
# sample_width: 1024
# sample_height: 1024
sample_scheduler: euler
attention_mechanism: sdpa
mixed_precision: float16
vae_dtype: auto
tf32: true
`,
};
