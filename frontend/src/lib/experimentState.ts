import { SamplingConfig } from "./defaultConfig.ts";
import { collapseStackToSingleLora, normalizeLegacySamplingParameters } from "./experimentUtils.ts";

const LEGACY_PLAYGROUND_STORAGE_KEY = "sampling-playground";
const LEGACY_SWEEP_STORAGE_KEY = "sampling-sweep";

export const EXPERIMENT_DRAFT_KEY = "experiment-draft";

export interface ExperimentDraftState {
  config: Record<string, unknown>;
}

export function defaultExperimentConfig(): Record<string, unknown> {
  return structuredClone(SamplingConfig.DEFAULT) as Record<string, unknown>;
}

export function defaultExperimentDraft(): ExperimentDraftState {
  return { config: defaultExperimentConfig() };
}

function migrateLegacySweepStorageKey(): void {
  if (typeof globalThis.localStorage === "undefined") return;
  if (globalThis.localStorage.getItem(EXPERIMENT_DRAFT_KEY)) return;
  const legacy = globalThis.localStorage.getItem(LEGACY_SWEEP_STORAGE_KEY);
  if (legacy) {
    globalThis.localStorage.setItem(EXPERIMENT_DRAFT_KEY, legacy);
  }
}

export function migrateLegacyExperimentFromPlayground(): void {
  if (typeof globalThis.localStorage === "undefined") return;
  migrateLegacySweepStorageKey();
  if (globalThis.localStorage.getItem(EXPERIMENT_DRAFT_KEY)) return;
  try {
    const raw = globalThis.localStorage.getItem(LEGACY_PLAYGROUND_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw) as { mode?: unknown; config?: unknown };
    if (parsed.mode !== "sweep" && parsed.mode !== "experiment") return;
    if (!parsed.config || typeof parsed.config !== "object") return;
    globalThis.localStorage.setItem(EXPERIMENT_DRAFT_KEY, JSON.stringify({ config: parsed.config }));
  } catch {
    // Ignore corrupt playground blobs; experiments fall back to defaults.
  }
}

export function loadExperimentDraft(): ExperimentDraftState {
  if (typeof globalThis.localStorage === "undefined") return defaultExperimentDraft();
  migrateLegacyExperimentFromPlayground();
  try {
    const raw = globalThis.localStorage.getItem(EXPERIMENT_DRAFT_KEY);
    if (!raw) return defaultExperimentDraft();
    const parsed = JSON.parse(raw) as Partial<ExperimentDraftState>;
    const defaults = defaultExperimentDraft();
    const config =
      parsed.config && typeof parsed.config === "object"
        ? { ...defaults.config, ...parsed.config }
        : defaults.config;
    return { config: normalizeLegacySamplingParameters(collapseStackToSingleLora(config)) };
  } catch {
    return defaultExperimentDraft();
  }
}

export function saveExperimentDraft(state: ExperimentDraftState): void {
  globalThis.localStorage.setItem(EXPERIMENT_DRAFT_KEY, JSON.stringify(state));
}
