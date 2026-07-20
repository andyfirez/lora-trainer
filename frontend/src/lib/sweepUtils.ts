export type SweepMode = "fixed" | "vary";

export interface SweepParameter {
  mode: SweepMode;
  value?: unknown;
  values?: unknown[];
}

export interface SweepParameters {
  base_model_name?: SweepParameter;
  lora_path?: SweepParameter;
  lora_weight?: SweepParameter;
  prompt?: SweepParameter;
  negative_prompt?: SweepParameter;
  steps?: SweepParameter;
  cfg_scale?: SweepParameter;
  width?: SweepParameter;
  height?: SweepParameter;
  scheduler?: SweepParameter;
  seed?: SweepParameter;
}

export interface GridLayout {
  x_axis?: string | null;
  y_axis?: string | null;
}

export const SWEEP_PARAM_ORDER = [
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
  "scheduler",
] as const;

export type SweepParamKey = (typeof SWEEP_PARAM_ORDER)[number];

export const SWEEP_PARAM_LABELS: Record<SweepParamKey, string> = {
  base_model_name: "Base Model",
  lora_path: "LoRA Path",
  lora_weight: "LoRA Weight",
  prompt: "Prompt",
  negative_prompt: "Negative Prompt",
  steps: "Steps",
  cfg_scale: "CFG Scale",
  width: "Width",
  height: "Height",
  scheduler: "Scheduler",
  seed: "Seed",
};

function effectiveValues(param: SweepParameter | undefined): unknown[] {
  if (!param) return [];
  if (param.mode === "vary" && param.values?.length) return param.values;
  if (param.value !== undefined && param.value !== null && param.value !== "") return [param.value];
  return [];
}

export function getParameters(config: Record<string, unknown>): SweepParameters {
  const raw = config.parameters;
  if (raw && typeof raw === "object") return raw as SweepParameters;
  const legacy: SweepParameters = {};
  const prompts = config.sample_prompts;
  if (Array.isArray(prompts)) {
    legacy.prompt =
      prompts.length > 1
        ? { mode: "vary", values: prompts }
        : { mode: "fixed", value: prompts[0] ?? "" };
  }
  if (config.base_model_name != null) {
    legacy.base_model_name = { mode: "fixed", value: config.base_model_name };
  }
  if (config.sample_negative_prompt != null) {
    legacy.negative_prompt = { mode: "fixed", value: config.sample_negative_prompt };
  }
  if (config.sample_steps != null) legacy.steps = { mode: "fixed", value: config.sample_steps };
  if (config.sample_cfg_scale != null) legacy.cfg_scale = { mode: "fixed", value: config.sample_cfg_scale };
  if (config.sample_width != null) legacy.width = { mode: "fixed", value: config.sample_width };
  if (config.sample_height != null) legacy.height = { mode: "fixed", value: config.sample_height };
  if (config.sample_scheduler != null) legacy.scheduler = { mode: "fixed", value: config.sample_scheduler };
  if (Array.isArray(config.lora_paths) && config.lora_paths.length) {
    legacy.lora_path = { mode: "vary", values: config.lora_paths };
  }
  return legacy;
}

export function varyKeysWithValues(parameters: SweepParameters): SweepParamKey[] {
  return SWEEP_PARAM_ORDER.filter((key) => {
    const param = parameters[key];
    return param?.mode === "vary" && (param.values?.length ?? 0) > 0;
  });
}

export function countCombinations(parameters: SweepParameters): number {
  const keys = varyKeysWithValues(parameters);
  if (!keys.length) {
    const prompts = effectiveValues(parameters.prompt);
    return prompts.filter((p) => String(p).trim()).length || 0;
  }
  return keys.reduce((acc, key) => acc * Math.max(effectiveValues(parameters[key]).length, 1), 1);
}

export interface GridPlanPreview {
  gridCount: number;
  rows: number;
  cols: number;
  sliceKeys: string[];
}

export function planGrids(parameters: SweepParameters, grid: GridLayout): GridPlanPreview {
  const varyKeys = varyKeysWithValues(parameters);
  if (varyKeys.length === 0) {
    const n = countCombinations(parameters);
    return { gridCount: n > 0 ? 0 : 0, rows: 1, cols: n, sliceKeys: [] };
  }
  if (varyKeys.length === 1) {
    const n = effectiveValues(parameters[varyKeys[0]]).length;
    return { gridCount: 1, rows: n, cols: 1, sliceKeys: [] };
  }
  const xAxis = grid.x_axis && varyKeys.includes(grid.x_axis as SweepParamKey) ? grid.x_axis : varyKeys[0];
  let yAxis =
    grid.y_axis && varyKeys.includes(grid.y_axis as SweepParamKey) ? grid.y_axis : varyKeys[1];
  if (xAxis === yAxis) yAxis = varyKeys.find((k) => k !== xAxis) ?? yAxis;
  const sliceKeys = varyKeys.filter((k) => k !== xAxis && k !== yAxis);
  const cols = effectiveValues(parameters[xAxis as SweepParamKey]).length || 1;
  const rows = effectiveValues(parameters[yAxis as SweepParamKey]).length || 1;
  let gridCount = 1;
  for (const key of sliceKeys) {
    gridCount *= effectiveValues(parameters[key]).length || 1;
  }
  return { gridCount, rows, cols, sliceKeys };
}

export function hasVaryingParamsExceptPrompt(parameters: SweepParameters): boolean {
  return SWEEP_PARAM_ORDER.some((key) => {
    if (key === "prompt") return false;
    const param = parameters[key];
    return param?.mode === "vary" && (param.values?.length ?? 0) > 1;
  });
}

export function sweepSummary(parameters: SweepParameters): string {
  const parts: string[] = [];
  for (const key of SWEEP_PARAM_ORDER) {
    const param = parameters[key];
    if (param?.mode === "vary" && (param.values?.length ?? 0) > 0) {
      parts.push(`${SWEEP_PARAM_LABELS[key]} (${param.values!.length})`);
    }
  }
  return parts.length ? parts.join(", ") : "all fixed";
}

export function defaultSweepParameter(type: "string" | "number" = "string"): SweepParameter {
  return { mode: "fixed", value: type === "number" ? 0 : "" };
}

export function setParameter(
  config: Record<string, unknown>,
  key: SweepParamKey,
  param: SweepParameter,
): Record<string, unknown> {
  const parameters = { ...getParameters(config), [key]: param };
  return { ...config, parameters };
}

export function setGridLayout(
  config: Record<string, unknown>,
  grid: GridLayout,
): Record<string, unknown> {
  return { ...config, grid };
}
