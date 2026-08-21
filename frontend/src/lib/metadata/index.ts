import { modelParameterMetadata } from "./model";
import { loraParameterMetadata } from "./lora";
import { trainingTargetsParameterMetadata } from "./trainingTargets";
import { trainingParameterMetadata } from "./training";
import { optimizerParameterMetadata } from "./optimizer";
import { dataParameterMetadata } from "./data";
import { optimizationParameterMetadata } from "./optimization";
import { performanceParameterMetadata } from "./performance";
import { checkpointingParameterMetadata } from "./checkpointing";
import { samplingParameterMetadata } from "./sampling";
import { loggingParameterMetadata } from "./logging";
import { buildParameterLookup } from "../parameterUtils";

export const TRAIN_PARAMETER_METADATA = [
  ...modelParameterMetadata,
  ...loraParameterMetadata,
  ...trainingTargetsParameterMetadata,
  ...trainingParameterMetadata,
  ...optimizerParameterMetadata,
  ...dataParameterMetadata,
  ...optimizationParameterMetadata,
  ...performanceParameterMetadata,
  ...checkpointingParameterMetadata,
  ...samplingParameterMetadata,
  ...loggingParameterMetadata,
];

export const TRAIN_PARAMETER_LOOKUP = buildParameterLookup(TRAIN_PARAMETER_METADATA);

export function trainHint(key: string): { hint?: string; hintAnchor?: string } {
  const meta = TRAIN_PARAMETER_LOOKUP.get(key);
  if (!meta || meta.showInlineHint === false) return {};
  return { hint: meta.shortHint, hintAnchor: meta.key };
}
