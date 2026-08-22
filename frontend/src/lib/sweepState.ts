import { SamplingConfig } from "./defaultConfig.ts";
import { collapseStackToSingleLora } from "./sweepUtils.ts";

const LEGACY_PLAYGROUND_STORAGE_KEY = "sampling-playground";

export const SWEEP_STORAGE_KEY = "sampling-sweep";

export interface SweepPersistedState {
  config: Record<string, unknown>;
}

export function defaultSweepConfig(): Record<string, unknown> {
  return structuredClone(SamplingConfig.DEFAULT) as Record<string, unknown>;
}

export function defaultSweepState(): SweepPersistedState {
  return { config: defaultSweepConfig() };
}

export function migrateLegacySweepFromPlayground(): void {
  if (typeof globalThis.localStorage === "undefined") return;
  if (globalThis.localStorage.getItem(SWEEP_STORAGE_KEY)) return;
  try {
    const raw = globalThis.localStorage.getItem(LEGACY_PLAYGROUND_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw) as { mode?: unknown; config?: unknown };
    if (parsed.mode !== "sweep") return;
    if (!parsed.config || typeof parsed.config !== "object") return;
    globalThis.localStorage.setItem(SWEEP_STORAGE_KEY, JSON.stringify({ config: parsed.config }));
  } catch {
    // Ignore corrupt playground blobs; sweep falls back to defaults.
  }
}

export function loadSweepState(): SweepPersistedState {
  if (typeof globalThis.localStorage === "undefined") return defaultSweepState();
  migrateLegacySweepFromPlayground();
  try {
    const raw = globalThis.localStorage.getItem(SWEEP_STORAGE_KEY);
    if (!raw) return defaultSweepState();
    const parsed = JSON.parse(raw) as Partial<SweepPersistedState>;
    const defaults = defaultSweepState();
    const config =
      parsed.config && typeof parsed.config === "object"
        ? { ...defaults.config, ...parsed.config }
        : defaults.config;
    return { config: collapseStackToSingleLora(config) };
  } catch {
    return defaultSweepState();
  }
}

export function saveSweepState(state: SweepPersistedState): void {
  globalThis.localStorage.setItem(SWEEP_STORAGE_KEY, JSON.stringify(state));
}
