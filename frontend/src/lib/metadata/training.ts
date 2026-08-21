import type { ParameterMeta } from "../parameterUtils";

export const trainingParameterMetadata: ParameterMeta[] = [
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
];
