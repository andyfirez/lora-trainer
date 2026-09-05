import { LEGACY_BUNDLED_SCHEDULERS } from "./sampleSamplerOptions.ts";

export type ExperimentMode = "fixed" | "vary";

export interface LoraEntry {
  path?: string | null;
  trigger?: string;
  weight?: number;
  bypassed?: boolean;
}

export interface ExperimentParameter {
  mode: ExperimentMode;
  value?: unknown;
  values?: unknown[];
}

export interface ExperimentParameters {
  base_model_name?: ExperimentParameter;
  lora_path?: ExperimentParameter;
  lora_weight?: ExperimentParameter;
  prompt?: ExperimentParameter;
  negative_prompt?: ExperimentParameter;
  steps?: ExperimentParameter;
  cfg_scale?: ExperimentParameter;
  width?: ExperimentParameter;
  height?: ExperimentParameter;
  sampler?: ExperimentParameter;
  scheduler?: ExperimentParameter;
  seed?: ExperimentParameter;
}

export interface GridLayout {
  x_axis?: string | null;
  y_axis?: string | null;
}

export const EXPERIMENT_PARAM_ORDER = [
  "lora_path",
  "lora_weight",
  "prompt",
  "cfg_scale",
  "steps",
  "seed",
  "base_model_name",
  "negative_prompt",
  "width",
  "height",
  "sampler",
  "scheduler",
] as const;

export type ExperimentParamKey = (typeof EXPERIMENT_PARAM_ORDER)[number];

export const EXPERIMENT_PARAM_LABELS: Record<ExperimentParamKey, string> = {
  base_model_name: "Base Model",
  lora_path: "LoRA Path",
  lora_weight: "LoRA Weight",
  prompt: "Prompt",
  negative_prompt: "Negative Prompt",
  steps: "Steps",
  cfg_scale: "CFG Scale",
  width: "Width",
  height: "Height",
  sampler: "Sampling method",
  scheduler: "Schedule type",
  seed: "Seed",
};

function effectiveValues(param: ExperimentParameter | undefined): unknown[] {
  if (!param) return [];
  if (param.mode === "vary" && param.values?.length) return param.values;
  if (param.value !== undefined && param.value !== null && param.value !== "") return [param.value];
  return [];
}

export function getParameters(config: Record<string, unknown>): ExperimentParameters {
  const raw = config.parameters;
  if (raw && typeof raw === "object") return raw as ExperimentParameters;
  return {};
}

export function emptyLoraEntry(): LoraEntry {
  return { path: null, trigger: "", weight: 1, bypassed: false };
}

function parseLoraWeight(value: unknown): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 1;
}

export function parseLoraEntry(value: unknown): LoraEntry {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const record = value as Record<string, unknown>;
    const path = record.path;
    const pathText = path == null ? null : String(path).trim();
    return {
      path: pathText || null,
      trigger: String(record.trigger ?? "").trim(),
      weight: parseLoraWeight(record.weight ?? 1),
      bypassed: Boolean(record.bypassed),
    };
  }
  if (value == null || (typeof value === "string" && !value.trim())) {
    return emptyLoraEntry();
  }
  return { path: String(value).trim(), trigger: "", weight: 1, bypassed: false };
}

export function parseLoraEntries(value: unknown): LoraEntry[] {
  if (Array.isArray(value)) {
    return value.length ? value.map(parseLoraEntry) : [emptyLoraEntry()];
  }
  return [parseLoraEntry(value)];
}

export function normalizeLoraEntry(entry: LoraEntry): LoraEntry {
  const path = entry.path == null ? null : String(entry.path).trim();
  return {
    path: path || null,
    trigger: String(entry.trigger ?? "").trim(),
    weight: parseLoraWeight(entry.weight ?? 1),
    bypassed: Boolean(entry.bypassed),
  };
}

function loraEntryPath(entry: LoraEntry): string | null {
  const path = entry.path == null ? null : String(entry.path).trim();
  return path || null;
}

function dedupeLoraEntries(entries: LoraEntry[]): LoraEntry[] {
  const result: LoraEntry[] = [];
  let seenEmpty = false;
  const seenFiles = new Set<string>();
  for (const entry of entries) {
    const path = loraEntryPath(entry);
    if (path == null) {
      if (!seenEmpty) {
        seenEmpty = true;
        result.push(normalizeLoraEntry(entry));
      }
      continue;
    }
    if (!seenFiles.has(path)) {
      seenFiles.add(path);
      result.push(normalizeLoraEntry(entry));
    }
  }
  return result;
}

export function loraEntriesFromParameter(param: ExperimentParameter | undefined): LoraEntry[] {
  if (!param) return [emptyLoraEntry()];
  if (param.mode === "vary" && param.values?.length) {
    return dedupeLoraEntries(param.values.map((v) => parseLoraEntry(v)));
  }
  if (param.value !== undefined && param.value !== null && param.value !== "") {
    return dedupeLoraEntries(parseLoraEntries(param.value));
  }
  return [emptyLoraEntry()];
}

export function loraParameterFromStack(entries: LoraEntry[]): ExperimentParameter {
  const withPath = entries.map(normalizeLoraEntry).filter((entry) => loraEntryPath(entry));
  if (withPath.length === 0) {
    return { mode: "fixed", value: emptyLoraEntry() };
  }
  if (withPath.length === 1) {
    return { mode: "fixed", value: withPath[0] };
  }
  return { mode: "fixed", value: withPath };
}

function effectiveValuesForKey(parameters: ExperimentParameters, key: ExperimentParamKey): unknown[] {
  const param = parameters[key];
  if (!param) return [];
  if (param.mode === "vary" && param.values?.length) {
    if (key === "lora_path") {
      return loraEntriesFromParameter(param).map((entry) => loraEntryPath(entry));
    }
    return param.values;
  }
  if (param.value !== undefined && param.value !== null && param.value !== "") {
    if (key === "lora_path") {
      return loraEntriesFromParameter(param).map((entry) => loraEntryPath(entry));
    }
    return [param.value];
  }
  if (key === "lora_path") return [null];
  return effectiveValues(param);
}

export function loraPathsFromParameter(param: ExperimentParameter | undefined): string[] {
  return loraEntriesFromParameter(param)
    .filter((entry) => !entry.bypassed)
    .map((entry) => loraEntryPath(entry))
    .filter((path): path is string => path != null);
}

export function syncLoraPathsToParameters(config: Record<string, unknown>): Record<string, unknown> {
  const parameters = { ...getParameters(config) };
  const loraPaths = loraPathsFromParameter(parameters.lora_path);
  return { ...config, parameters, lora_paths: loraPaths };
}

export function varyKeysWithValues(parameters: ExperimentParameters): ExperimentParamKey[] {
  return EXPERIMENT_PARAM_ORDER.filter((key) => {
    const param = parameters[key];
    return param?.mode === "vary" && (param.values?.length ?? 0) > 0;
  });
}

export function countCombinations(parameters: ExperimentParameters): number {
  const keys = varyKeysWithValues(parameters);
  if (!keys.length) {
    const prompts = effectiveValues(parameters.prompt);
    return prompts.filter((p) => String(p).trim()).length || 0;
  }
  return keys.reduce(
    (acc, key) => acc * Math.max(effectiveValuesForKey(parameters, key).length, 1),
    1,
  );
}

export interface GridPlanPreview {
  gridCount: number;
  rows: number;
  cols: number;
  sliceKeys: string[];
}

export function planGrids(parameters: ExperimentParameters, grid: GridLayout): GridPlanPreview {
  const varyKeys = varyKeysWithValues(parameters);
  if (varyKeys.length === 0) {
    const n = countCombinations(parameters);
    return { gridCount: n > 0 ? 0 : 0, rows: 1, cols: n, sliceKeys: [] };
  }
  if (varyKeys.length === 1) {
    const n = effectiveValues(parameters[varyKeys[0]]).length;
    return { gridCount: 1, rows: n, cols: 1, sliceKeys: [] };
  }
  const xAxis = grid.x_axis && varyKeys.includes(grid.x_axis as ExperimentParamKey) ? grid.x_axis : varyKeys[0];
  let yAxis =
    grid.y_axis && varyKeys.includes(grid.y_axis as ExperimentParamKey) ? grid.y_axis : varyKeys[1];
  if (xAxis === yAxis) yAxis = varyKeys.find((k) => k !== xAxis) ?? yAxis;
  const sliceKeys = varyKeys.filter((k) => k !== xAxis && k !== yAxis);
  const cols = effectiveValues(parameters[xAxis as ExperimentParamKey]).length || 1;
  const rows = effectiveValues(parameters[yAxis as ExperimentParamKey]).length || 1;
  let gridCount = 1;
  for (const key of sliceKeys) {
    gridCount *= effectiveValues(parameters[key]).length || 1;
  }
  return { gridCount, rows, cols, sliceKeys };
}

export function hasVaryingParamsExceptPrompt(parameters: ExperimentParameters): boolean {
  return EXPERIMENT_PARAM_ORDER.some((key) => {
    if (key === "prompt") return false;
    const param = parameters[key];
    return param?.mode === "vary" && (param.values?.length ?? 0) > 1;
  });
}

export function experimentSummary(parameters: ExperimentParameters): string {
  const parts: string[] = [];
  for (const key of EXPERIMENT_PARAM_ORDER) {
    const param = parameters[key];
    if (param?.mode === "vary" && (param.values?.length ?? 0) > 0) {
      parts.push(`${EXPERIMENT_PARAM_LABELS[key]} (${param.values!.length})`);
    }
  }
  return parts.length ? parts.join(", ") : "all fixed";
}

export function defaultExperimentParameter(type: "string" | "number" = "string"): ExperimentParameter {
  return { mode: "fixed", value: type === "number" ? 0 : "" };
}

export function setParameter(
  config: Record<string, unknown>,
  key: ExperimentParamKey,
  param: ExperimentParameter,
): Record<string, unknown> {
  const parameters = { ...getParameters(config), [key]: param };
  return { ...config, parameters };
}

export function clearPlaygroundLoraState(config: Record<string, unknown>): Record<string, unknown> {
  let next = setParameter(config, "lora_path", { mode: "fixed", value: emptyLoraEntry() });
  next = setParameter(next, "lora_weight", { mode: "fixed", value: 1 });
  return { ...next, lora_paths: [] };
}

export function applyLoraStack(config: Record<string, unknown>, entries: LoraEntry[]): Record<string, unknown> {
  const normalized = entries.map(normalizeLoraEntry);
  const first = normalized.find((entry) => loraEntryPath(entry) && !entry.bypassed);
  const value = normalized.length <= 1 ? (normalized[0] ?? emptyLoraEntry()) : normalized;
  let next = setParameter(config, "lora_path", { mode: "fixed", value });
  if (first) {
    next = setParameter(next, "lora_weight", { mode: "fixed", value: first.weight ?? 1 });
  }
  return syncLoraPathsToParameters(next);
}

export function collapseStackToSingleLora(config: Record<string, unknown>): Record<string, unknown> {
  const param = getParameters(config).lora_path;
  if (!param || param.mode === "vary") return syncLoraPathsToParameters(config);
  const first = parseLoraEntries(param.value).find((entry) => loraEntryPath(entry)) ?? emptyLoraEntry();
  return applyLoraStack(config, [first]);
}

export function setGridLayout(
  config: Record<string, unknown>,
  grid: GridLayout,
): Record<string, unknown> {
  return { ...config, grid };
}

function paramValues(param: ExperimentParameter | undefined): unknown[] {
  if (!param) return [];
  if (param.mode === "vary" && param.values?.length) return param.values;
  if (param.value !== undefined && param.value !== null) return [param.value];
  return [];
}

export function normalizeLegacySamplingParameters(
  config: Record<string, unknown>,
): Record<string, unknown> {
  const parameters = getParameters(config);
  const samplerExplicit = parameters.sampler != null;
  const schedulerValues = paramValues(parameters.scheduler);
  if (!schedulerValues.length) return config;
  const mapped = schedulerValues.map((value) => LEGACY_BUNDLED_SCHEDULERS[String(value)]);
  if (mapped.some((item) => item == null)) return config;
  const samplers = mapped.map((item) => item!.sampler);
  const schedules = mapped.map((item) => item!.scheduler);
  let next = config;
  if (!samplerExplicit) {
    next = setParameter(next, "sampler", {
      mode: parameters.scheduler?.mode === "vary" && samplers.length > 1 ? "vary" : "fixed",
      ...(samplers.length > 1 ? { values: samplers } : { value: samplers[0] }),
    });
    next = setParameter(next, "scheduler", {
      mode: parameters.scheduler?.mode === "vary" && schedules.length > 1 ? "vary" : "fixed",
      ...(schedules.length > 1 ? { values: schedules } : { value: schedules[0] }),
    });
    return next;
  }
  const samplerValues = paramValues(parameters.sampler);
  if (
    samplerValues.length === 0 ||
    (samplerValues.length === samplers.length && samplerValues.every((value, index) => value === samplers[index]))
  ) {
    next = setParameter(next, "scheduler", {
      mode: parameters.scheduler?.mode === "vary" && schedules.length > 1 ? "vary" : "fixed",
      ...(schedules.length > 1 ? { values: schedules } : { value: schedules[0] }),
    });
  }
  return next;
}
