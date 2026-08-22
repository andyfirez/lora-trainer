import assert from "node:assert/strict";
import test from "node:test";

import {
  PNG_INFO_STORAGE_KEY,
  defaultPngInspectPersistedState,
  loadPngInspectState,
  savePngInspectState,
} from "./pngInfoState.ts";

test("loadPngInspectState returns defaults when storage is empty", () => {
  const storage = new Map<string, string>();
  const original = globalThis.sessionStorage;
  Object.defineProperty(globalThis, "sessionStorage", {
    configurable: true,
    value: {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => {
        storage.set(key, value);
      },
      removeItem: (key: string) => {
        storage.delete(key);
      },
    },
  });
  try {
    assert.deepEqual(loadPngInspectState(), defaultPngInspectPersistedState());
  } finally {
    Object.defineProperty(globalThis, "sessionStorage", {
      configurable: true,
      value: original,
    });
  }
});

test("savePngInspectState persists and reloads inspect result", () => {
  const storage = new Map<string, string>();
  const original = globalThis.sessionStorage;
  Object.defineProperty(globalThis, "sessionStorage", {
    configurable: true,
    value: {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => {
        storage.set(key, value);
      },
      removeItem: (key: string) => {
        storage.delete(key);
      },
    },
  });
  try {
    savePngInspectState({
      fileName: "sample.png",
      result: {
        info: "Steps: 20",
        items: { parameters: "Steps: 20" },
        parameters: { Steps: 20 },
        width: 512,
        height: 768,
        preview_base64: "data:image/png;base64,abc",
      },
      error: null,
    });
    assert.ok(storage.has(PNG_INFO_STORAGE_KEY));
    const loaded = loadPngInspectState();
    assert.equal(loaded.fileName, "sample.png");
    assert.equal(loaded.result?.width, 512);
  } finally {
    Object.defineProperty(globalThis, "sessionStorage", {
      configurable: true,
      value: original,
    });
  }
});
