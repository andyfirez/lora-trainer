import type { ParameterMeta } from "../parameterUtils";

export const loggingParameterMetadata: ParameterMeta[] = [
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
