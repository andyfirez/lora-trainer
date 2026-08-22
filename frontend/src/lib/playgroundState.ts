import { SamplingConfig } from "./defaultConfig.ts";
import { migrateLegacySweepFromPlayground } from "./sweepState.ts";
import {
  SWEEP_PARAM_ORDER,
  getParameters,
  setParameter,
  syncLoraPathsToParameters,
  type SweepParamKey,
} from "./sweepUtils.ts";

export const PLAYGROUND_STORAGE_KEY = "sampling-playground";

export interface PlaygroundPersistedState {
  config: Record<string, unknown>;
  batchCount: number;
}

export function defaultPlaygroundConfig(): Record<string, unknown> {
  return structuredClone(SamplingConfig.DEFAULT) as Record<string, unknown>;
}

export function defaultPlaygroundState(): PlaygroundPersistedState {
  return {
    config: defaultPlaygroundConfig(),
    batchCount: 1,
  };
}

export function loadPlaygroundState(): PlaygroundPersistedState {
  if (typeof globalThis.localStorage === "undefined") return defaultPlaygroundState();
  migrateLegacySweepFromPlayground();
  try {
    const raw = globalThis.localStorage.getItem(PLAYGROUND_STORAGE_KEY);
    if (!raw) return defaultPlaygroundState();
    const parsed = JSON.parse(raw) as Partial<PlaygroundPersistedState>;
    const defaults = defaultPlaygroundState();
    const merged =
      parsed.config && typeof parsed.config === "object"
        ? { ...defaults.config, ...parsed.config }
        : defaults.config;
    return {
      config: applySamplingSizeDefaults(collapseSweepToFixed(merged)),
      batchCount:
        typeof parsed.batchCount === "number" && Number.isFinite(parsed.batchCount) && parsed.batchCount >= 1
          ? Math.min(32, Math.floor(parsed.batchCount))
          : 1,
    };
  } catch {
    return defaultPlaygroundState();
  }
}

export function savePlaygroundState(state: PlaygroundPersistedState): void {
  globalThis.localStorage.setItem(PLAYGROUND_STORAGE_KEY, JSON.stringify(state));
}

export function collapseSweepToFixed(config: Record<string, unknown>): Record<string, unknown> {
  let next = { ...config };
  for (const key of SWEEP_PARAM_ORDER) {
    const param = getParameters(next)[key as SweepParamKey];
    if (param?.mode !== "vary") continue;
    next = setParameter(next, key, {
      mode: "fixed",
      value: param.values?.[0] ?? param.value ?? null,
    });
  }
  return syncLoraPathsToParameters(next);
}

function applySamplingSizeDefaults(config: Record<string, unknown>): Record<string, unknown> {
  const parameters = getParameters(config);
  let next = config;
  const width = parameters.width?.value;
  const height = parameters.height?.value;
  if (width == null || width === "" || Number.isNaN(Number(width))) {
    next = setParameter(next, "width", { mode: "fixed", value: 832 });
  }
  if (height == null || height === "" || Number.isNaN(Number(height))) {
    next = setParameter(next, "height", { mode: "fixed", value: 1216 });
  }
  return next;
}

export function applyBatchCount(
  config: Record<string, unknown>,
  batchCount: number,
  randomSeed: () => number,
): Record<string, unknown> {
  if (batchCount <= 1) return config;
  const seedParam = getParameters(config).seed;
  const current = seedParam?.mode === "vary" ? seedParam.values?.[0] : seedParam?.value;
  const base = current == null || current === "" || Number.isNaN(Number(current)) ? randomSeed() : Number(current);
  const values = Array.from({ length: batchCount }, (_, index) => base + index);
  return setParameter(config, "seed", { mode: "vary", values });
}

export function validatePlaygroundConfig(config: Record<string, unknown>): string | null {
  const prompt = getParameters(config).prompt;
  const values = prompt?.mode === "vary" ? (prompt.values ?? []) : [prompt?.value];
  if (!values.some((value) => String(value ?? "").trim())) return "Prompt is required";
  return null;
}

export function randomSeed(): number {
  return Math.floor(Math.random() * 2 ** 31);
}

export function formatParamsInfotext(params: Record<string, unknown>): string {
  const prompt = String(params.prompt ?? "").trim();
  const negative = String(params.negative_prompt ?? "").trim();
  const lines = [prompt];
  if (negative) lines.push(`Negative prompt: ${negative}`);
  const width = params.width ?? 832;
  const height = params.height ?? 1216;
  const fields = [
    `Steps: ${params.steps ?? 30}`,
    `Sampler: ${params.scheduler ?? "euler"}`,
    `CFG scale: ${params.cfg_scale ?? 7.5}`,
    `Seed: ${params.seed ?? -1}`,
    `Size: ${width}x${height}`,
  ];
  const stack = params.lora_stack;
  if (Array.isArray(stack) && stack.length > 1) {
    const formatted = stack
      .map((item) => {
        const record = item as { path?: unknown; weight?: unknown };
        const name = String(record.path ?? "").split(/[/\\]/).pop() ?? "";
        return `${name} (${record.weight ?? 1})`;
      })
      .filter(Boolean)
      .join(", ");
    if (formatted) fields.push(`Lora: ${formatted}`);
  } else {
    if (params.lora_path) fields.push(`Lora: ${String(params.lora_path)}`);
    if (params.lora_weight != null && params.lora_path) fields.push(`Lora weight: ${params.lora_weight}`);
  }
  lines.push(fields.join(", "));
  return lines.filter(Boolean).join("\n");
}

export function seedFromParams(params: Record<string, unknown> | undefined): number | null {
  if (params == null || params.seed == null || params.seed === "") return null;
  const value = Number(params.seed);
  return Number.isFinite(value) ? value : null;
}
