import type { ParameterMeta } from "../parameterUtils";

export const checkpointingParameterMetadata: ParameterMeta[] = [
// Checkpointing
  {
    key: "checkpointing_enabled",
    label: "Enable Checkpoints",
    section: "Checkpointing",
    shortHint: "Save intermediate LoRA weights during training.",
    description:
      "When enabled, saves LoRA checkpoints at regular epoch intervals. Required for post-train auto sampling and resume.",
    defaultValue: "true",
    showInlineHint: false,
    recommendedValue: "true",
    valueOptions: [
      { value: "true", description: "Save intermediate weights; required for post-train sampling and resume." },
      { value: "false", description: "Only save final LoRA; disables post-train auto sampling." },
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
];
