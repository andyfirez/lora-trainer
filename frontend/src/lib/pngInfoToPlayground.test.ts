import assert from "node:assert/strict";
import test from "node:test";

import { PLAYGROUND_STORAGE_KEY } from "./playgroundState.ts";
import { applyPngInfoToPlayground } from "./pngInfoToPlayground.ts";

test("applyPngInfoToPlayground maps parsed PNG parameters to playground config", () => {
  const storage = new Map<string, string>();
  const original = globalThis.localStorage;
  Object.defineProperty(globalThis, "localStorage", {
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
    const next = applyPngInfoToPlayground(
      {
        Prompt: "a cat in a hat",
        "Negative prompt": "blurry, low quality",
        Steps: "20",
        Sampler: "Euler a",
        "CFG scale": "7",
        Seed: "965400086",
        "Size-1": 512,
        "Size-2": 512,
        Model: "stable-diffusion-xl-base-1.0",
      },
      { width: 64, height: 64 },
    );

    const parameters = next.config.parameters as Record<
      string,
      { mode?: string; value?: unknown }
    >;
    assert.equal(parameters.prompt?.value, "a cat in a hat");
    assert.equal(parameters.negative_prompt?.value, "blurry, low quality");
    assert.equal(parameters.steps?.value, 20);
    assert.equal(parameters.cfg_scale?.value, 7);
    assert.equal(parameters.seed?.value, 965400086);
    assert.equal(parameters.width?.value, 512);
    assert.equal(parameters.height?.value, 512);
    assert.equal(parameters.scheduler?.value, "euler_a");
    assert.equal(parameters.base_model_name?.value, "stable-diffusion-xl-base-1.0");
  } finally {
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: original,
    });
  }
});

test("applyPngInfoToPlayground falls back to image dimensions when size is missing", () => {
  const storage = new Map<string, string>();
  const original = globalThis.localStorage;
  Object.defineProperty(globalThis, "localStorage", {
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
    storage.set(PLAYGROUND_STORAGE_KEY, JSON.stringify({ config: {}, batchCount: 1 }));
    const next = applyPngInfoToPlayground({ Prompt: "portrait" }, { width: 832, height: 1216 });
    const parameters = next.config.parameters as Record<string, { value?: unknown }>;
    assert.equal(parameters.prompt?.value, "portrait");
    assert.equal(parameters.width?.value, 832);
    assert.equal(parameters.height?.value, 1216);
  } finally {
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: original,
    });
  }
});
