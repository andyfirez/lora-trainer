import type { ParameterMeta } from "../parameterUtils";

export const optimizerParameterMetadata: ParameterMeta[] = [
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
];
