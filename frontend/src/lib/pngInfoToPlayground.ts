import { setParameter } from "./experimentUtils.ts";
import { loadPlaygroundState, type PlaygroundPersistedState } from "./playgroundState.ts";

const A1111_SAMPLER_TO_SCHEDULER: Record<string, string> = {
  euler: "euler",
  "euler a": "euler_a",
  euler_a: "euler_a",
  ddim: "ddim",
  "dpm++ 2m": "dpm++",
  "dpm++": "dpm++",
};

function parseScheduler(value: string | number | undefined): string | null {
  if (value == null) return null;
  const normalized = String(value).trim().toLowerCase();
  return A1111_SAMPLER_TO_SCHEDULER[normalized] ?? null;
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

  const scheduler = parseScheduler(parameters.Sampler);
  if (scheduler) {
    config = setParameter(config, "scheduler", { mode: "fixed", value: scheduler });
  }

  const model = parseString(parameters.Model);
  if (model) {
    config = setParameter(config, "base_model_name", { mode: "fixed", value: model });
  }

  return { ...state, config };
}
