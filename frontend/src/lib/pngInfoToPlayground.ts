import { setParameter } from "./experimentUtils.ts";
import { loadPlaygroundState, type PlaygroundPersistedState } from "./playgroundState.ts";
import { LEGACY_BUNDLED_SCHEDULERS } from "./sampleSamplerOptions.ts";

const SAMPLER_ALIASES: Record<string, string> = {
  euler: "euler",
  "euler a": "euler_ancestral",
  euler_a: "euler_ancestral",
  euler_ancestral: "euler_ancestral",
  ddim: "ddim",
  "dpm++ 2m": "dpmpp_2m",
  "dpm++": "dpmpp_2m",
  dpmpp_2m: "dpmpp_2m",
  heun: "heun",
  lms: "lms",
  "dpm++ 2s a": "dpmpp_2s_ancestral",
  dpmpp_2s_ancestral: "dpmpp_2s_ancestral",
  "dpm++ sde": "dpmpp_sde",
  dpmpp_sde: "dpmpp_sde",
};

const SCHEDULER_ALIASES: Record<string, string> = {
  normal: "normal",
  karras: "karras",
  exponential: "exponential",
  simple: "simple",
  "sgm uniform": "sgm_uniform",
  sgm_uniform: "sgm_uniform",
  ddim: "ddim_uniform",
  ddim_uniform: "ddim_uniform",
  beta: "beta",
  "linear quadratic": "linear_quadratic",
  linear_quadratic: "linear_quadratic",
  "kl optimal": "kl_optimal",
  kl_optimal: "kl_optimal",
  automatic: "automatic",
};

function parseSamplerAndScheduler(
  samplerField: string | number | undefined,
  scheduleField: string | number | undefined,
): { sampler: string; scheduler: string } | null {
  if (samplerField == null && scheduleField == null) return null;
  const samplerText = samplerField == null ? "" : String(samplerField).trim().toLowerCase();
  const scheduleText = scheduleField == null ? "" : String(scheduleField).trim().toLowerCase();
  const legacy = LEGACY_BUNDLED_SCHEDULERS[samplerText];
  let sampler = SAMPLER_ALIASES[samplerText] ?? legacy?.sampler ?? null;
  let scheduler = SCHEDULER_ALIASES[scheduleText] ?? null;
  if (sampler == null && samplerText) {
    for (const suffix of Object.keys(SCHEDULER_ALIASES).sort((a, b) => b.length - a.length)) {
      if (samplerText.endsWith(` ${suffix}`)) {
        sampler = SAMPLER_ALIASES[samplerText.slice(0, -(suffix.length + 1)).trim()] ?? null;
        scheduler = scheduler ?? SCHEDULER_ALIASES[suffix];
        break;
      }
    }
  }
  if (sampler == null && scheduleField == null && legacy) {
    return { sampler: legacy.sampler, scheduler: legacy.scheduler };
  }
  if (sampler == null) return null;
  return { sampler, scheduler: scheduler ?? legacy?.scheduler ?? "normal" };
}

function parseNumber(value: string | number | undefined): number | null {
  if (value == null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseString(value: string | number | undefined): string | null {
  if (value == null) return null;
  const text = String(value).trim();
  return text || null;
}

export function applyPngInfoToPlayground(
  parameters: Record<string, string | number>,
  imageSize?: { width: number; height: number },
): PlaygroundPersistedState {
  const state = loadPlaygroundState();
  let config = state.config;

  const prompt = parseString(parameters.Prompt);
  if (prompt) {
    config = setParameter(config, "prompt", { mode: "fixed", value: prompt });
  }

  const negativePrompt = parseString(parameters["Negative prompt"]);
  if (negativePrompt) {
    config = setParameter(config, "negative_prompt", { mode: "fixed", value: negativePrompt });
  }

  const steps = parseNumber(parameters.Steps);
  if (steps != null) {
    config = setParameter(config, "steps", { mode: "fixed", value: steps });
  }

  const cfgScale = parseNumber(parameters["CFG scale"]);
  if (cfgScale != null) {
    config = setParameter(config, "cfg_scale", { mode: "fixed", value: cfgScale });
  }

  const seed = parseNumber(parameters.Seed);
  if (seed != null && seed >= 0) {
    config = setParameter(config, "seed", { mode: "fixed", value: seed });
  }

  const width = parseNumber(parameters["Size-1"]) ?? imageSize?.width ?? null;
  const height = parseNumber(parameters["Size-2"]) ?? imageSize?.height ?? null;
  if (width != null) {
    config = setParameter(config, "width", { mode: "fixed", value: width });
  }
  if (height != null) {
    config = setParameter(config, "height", { mode: "fixed", value: height });
  }

  const parsed = parseSamplerAndScheduler(parameters.Sampler, parameters["Schedule type"]);
  if (parsed) {
    config = setParameter(config, "sampler", { mode: "fixed", value: parsed.sampler });
    config = setParameter(config, "scheduler", { mode: "fixed", value: parsed.scheduler });
  }

  const model = parseString(parameters.Model);
  if (model) {
    config = setParameter(config, "base_model_name", { mode: "fixed", value: model });
  }

  return { ...state, config };
}
