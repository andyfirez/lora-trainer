import type { ParameterMeta } from "../parameterUtils";

export const modelParameterMetadata: ParameterMeta[] = [
// Model
  {
    key: "base_model_name",
    label: "Base Model",
    section: "Model",
    shortHint: "Local SDXL checkpoint or diffusers folder from the base models directory in Settings.",
    description:
      "Specifies the SDXL base model to fine-tune. Choose a folder or checkpoint file from the base models root configured in Settings. The model architecture must match SDXL.",
    defaultValue: "",
    showInlineHint: false,
    recommendedValue: "",
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
];
