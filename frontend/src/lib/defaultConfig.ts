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
  learning_rate: 0.00005
text_encoder_1:
  train: false
  weight_dtype: float16
  learning_rate: 0.00005
text_encoder_2:
  train: false
  weight_dtype: float16
  learning_rate: 0.00005

# Training hyperparameters
epochs: 30
batch_size: 1
gradient_accumulation_steps: 1
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
output_dir: output
source_type: manual
lora_paths: []
include_final_checkpoint: true
include_base_model_sample: false

grid:
  x_axis: null
  y_axis: null

parameters:
  base_model_name:
    mode: fixed
    value: stabilityai/stable-diffusion-xl-base-1.0
  lora_path:
    mode: fixed
    value:
      path: null
      trigger: ""
  lora_weight:
    mode: fixed
    value: 1.0
  prompt:
    mode: fixed
    value: ""
  negative_prompt:
    mode: fixed
    value: ""
  steps:
    mode: fixed
    value: 30
  cfg_scale:
    mode: fixed
    value: 7.5
  width:
    mode: fixed
    value: null
  height:
    mode: fixed
    value: null
  scheduler:
    mode: fixed
    value: euler
  seed:
    mode: fixed
    value: null

attention_mechanism: sdpa
mixed_precision: float16
vae_dtype: auto
tf32: true
sample_vae_tiling: true
`,
};
