import presets from "./optimizer_presets.json";

export type OptimizerType = "adamw" | "adamw_8bit" | "adafactor" | "prodigy";

export interface OptimizerBlock {
  type: OptimizerType;
  weight_decay: number;
  beta1: number;
  beta2: number;
  relative_step: boolean;
  scale_parameter: boolean;
  warmup_init: boolean;
  decouple: boolean;
  use_bias_correction: boolean;
  safeguard_warmup: boolean;
  d0: number;
  d_coef: number;
}

export interface OptimizerPreset {
  optimizer: OptimizerBlock;
  learning_rate: number;
  lr_scheduler: string;
  lr_warmup_steps: number;
}

type Config = Record<string, unknown>;

const optimizerPresets = presets as Record<OptimizerType, OptimizerPreset>;

export function getOptimizerPreset(type: OptimizerType): OptimizerPreset {
  return optimizerPresets[type];
}

export function applyOptimizerPreset(config: Config, type: OptimizerType): Config {
  const preset = getOptimizerPreset(type);
  return {
    ...config,
    optimizer: { ...preset.optimizer },
    learning_rate: preset.learning_rate,
    lr_scheduler: preset.lr_scheduler,
    lr_warmup_steps: preset.lr_warmup_steps,
  };
}

export const optimizerOptions = [
  { value: "adamw", label: "AdamW" },
  { value: "adamw_8bit", label: "AdamW 8-bit" },
  { value: "adafactor", label: "Adafactor" },
  { value: "prodigy", label: "Prodigy" },
] as const;
