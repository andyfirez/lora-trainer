import type { PngInfoResponse } from "@/types";

export const PNG_INFO_STORAGE_KEY = "png-info-inspect";

export interface PersistedPngInspectState {
  fileName: string | null;
  result: PngInfoResponse | null;
  error: string | null;
}

export function defaultPngInspectPersistedState(): PersistedPngInspectState {
  return {
    fileName: null,
    result: null,
    error: null,
  };
}

export function loadPngInspectState(): PersistedPngInspectState {
  if (typeof globalThis.sessionStorage === "undefined") {
    return defaultPngInspectPersistedState();
  }
  try {
    const raw = globalThis.sessionStorage.getItem(PNG_INFO_STORAGE_KEY);
    if (!raw) return defaultPngInspectPersistedState();
    const parsed = JSON.parse(raw) as Partial<PersistedPngInspectState>;
    return {
      fileName: typeof parsed.fileName === "string" ? parsed.fileName : null,
      result: parsed.result && typeof parsed.result === "object" ? parsed.result : null,
      error: typeof parsed.error === "string" ? parsed.error : null,
    };
  } catch {
    return defaultPngInspectPersistedState();
  }
}

export function savePngInspectState(state: PersistedPngInspectState): void {
  if (typeof globalThis.sessionStorage === "undefined") return;
  if (!state.result && !state.error) {
    globalThis.sessionStorage.removeItem(PNG_INFO_STORAGE_KEY);
    return;
  }
  globalThis.sessionStorage.setItem(PNG_INFO_STORAGE_KEY, JSON.stringify(state));
}
