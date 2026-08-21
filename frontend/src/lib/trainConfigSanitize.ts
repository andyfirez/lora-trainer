import { applyOptimizerPreset, type OptimizerType } from "@/lib/optimizerPresets";
import type { Dataset } from "@/types";

export type TrainConfig = Record<string, unknown>;

const INLINE_SAMPLING_KEYS = [
  "sample_prompts",
  "sample_negative_prompt",
  "sample_steps",
  "sample_cfg_scale",
  "sample_width",
  "sample_height",
  "sample_scheduler",
  "post_training_sampling_config_id",
  "sample_after_training",
] as const;

const LEGACY_CONCEPT_KEYS = ["image_dir"] as const;

export function stripLoraVersionSuffix(name: string): string {
  return name.replace(/_v\d+$/, "");
}

function normalizeConcept(concept: unknown, datasets?: Dataset[]): TrainConfig {
  if (!concept || typeof concept !== "object") return concept as TrainConfig;
  const raw = concept as TrainConfig;
  let datasetId = raw.dataset_id as number | null | undefined;
  const legacyDir = raw.image_dir as string | undefined;

  if (datasetId == null && legacyDir && datasets?.length) {
    datasetId = datasets.find((d) => d.relative_path === legacyDir || d.resolved_path === legacyDir)?.id;
  }
  if (datasetId == null && datasets?.length) {
    datasetId = datasets[0].id;
  }

  const item = { ...raw };
  if (datasetId != null) {
    item.dataset_id = datasetId;
  }
  if (item.dataset_id != null) {
    for (const key of LEGACY_CONCEPT_KEYS) {
      delete item[key];
    }
  }
  return item;
}

export function isTextEncoderTrainingEnabled(config: TrainConfig): boolean {
  const te1 = config.text_encoder_1 as TrainConfig | undefined;
  const te2 = config.text_encoder_2 as TrainConfig | undefined;
  return Boolean(te1?.train || te2?.train);
}

function stripInlineSamplingFields(next: TrainConfig): TrainConfig {
  const cleaned = { ...next };
  for (const key of INLINE_SAMPLING_KEYS) {
    delete cleaned[key];
  }
  delete cleaned.sampling_enabled;
  delete cleaned.sampling_config_id;
  return cleaned;
}

export function sanitizeTrainConfig(next: TrainConfig, datasets?: Dataset[]): TrainConfig {
  let cleaned = stripInlineSamplingFields(next);
  if (cleaned.clip_skip == null) {
    cleaned = { ...cleaned, clip_skip: 2 };
  }
  if (isTextEncoderTrainingEnabled(cleaned)) {
    cleaned = {
      ...cleaned,
      cache_text_encoder_outputs: false,
      cache_text_encoder_outputs_to_disk: false,
    };
  }
  const concepts = cleaned.concepts;
  if (Array.isArray(concepts)) {
    cleaned = {
      ...cleaned,
      concepts: concepts.map((concept) => normalizeConcept(concept, datasets)),
    };
  }
  return cleaned;
}

export function applyTrainConfigPatch(
  config: TrainConfig,
  key: string,
  value: unknown,
  datasets?: Dataset[],
): TrainConfig {
  let next: TrainConfig = { ...config };
  if (value === undefined) {
    delete next[key];
  } else {
    next = { ...next, [key]: value };
  }
  if (key === "cache_latents" && value === false) {
    next = { ...next, cache_latents_to_disk: false };
  }
  if (key === "cache_text_encoder_outputs" && value === false) {
    next = { ...next, cache_text_encoder_outputs_to_disk: false };
  }
  return sanitizeTrainConfig(next, datasets);
}

export function applyTrainConfigNestedPatch(
  config: TrainConfig,
  parent: string,
  key: string,
  value: unknown,
  datasets?: Dataset[],
): TrainConfig {
  const parentValue = (config[parent] ?? {}) as TrainConfig;
  return sanitizeTrainConfig(
    {
      ...config,
      [parent]: { ...parentValue, [key]: value },
    },
    datasets,
  );
}

export function applyTrainOptimizerType(
  config: TrainConfig,
  type: OptimizerType,
  datasets?: Dataset[],
): TrainConfig {
  return sanitizeTrainConfig(applyOptimizerPreset(config, type), datasets);
}

export function getOptimizerType(config: TrainConfig): OptimizerType {
  const optimizer = config.optimizer as TrainConfig | undefined;
  return (optimizer?.type as OptimizerType) ?? "adamw_8bit";
}

export function isDatasetCompatibleWithTrain(
  dataset: Dataset,
  trainResolution: number,
  trainEnableBucket: boolean,
): boolean {
  if (!dataset.preprocess_ready || dataset.target_resolution !== trainResolution) {
    return false;
  }
  return Boolean(dataset.enable_bucket) === trainEnableBucket;
}

export function datasetOptionLabel(
  dataset: Dataset,
  trainResolution: number,
  trainEnableBucket: boolean,
): string {
  if (dataset.target_resolution == null) {
    return `${dataset.name} (no target resolution)`;
  }
  if (dataset.target_resolution !== trainResolution) {
    return `${dataset.name} (${dataset.target_resolution}px ≠ ${trainResolution}px)`;
  }
  if (!dataset.preprocess_ready) {
    return `${dataset.name} (not prepared)`;
  }
  if (Boolean(dataset.enable_bucket) !== trainEnableBucket) {
    return `${dataset.name} (bucket ${dataset.enable_bucket ? "on" : "off"} ≠ train)`;
  }
  return dataset.name;
}
