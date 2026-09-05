import { strict as assert } from "node:assert";
import { afterEach, beforeEach, describe, it } from "node:test";

import {
  EXPERIMENT_DRAFT_KEY,
  loadExperimentDraft,
} from "./experimentState.ts";
import { PLAYGROUND_STORAGE_KEY, collapseExperimentToFixed, loadPlaygroundState } from "./playgroundState.ts";
import { applyLoraStack, getParameters, setParameter } from "./experimentUtils.ts";

function installLocalStorage() {
  const store = new Map<string, string>();
  const localStorage = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, String(value));
    },
    removeItem: (key: string) => {
      store.delete(key);
    },
    clear: () => store.clear(),
  };
  Object.defineProperty(globalThis, "localStorage", {
    value: localStorage,
    configurable: true,
  });
  return store;
}

describe("playground and experiment persisted state", () => {
  beforeEach(() => {
    installLocalStorage();
  });

  afterEach(() => {
    delete (globalThis as { localStorage?: unknown }).localStorage;
  });

  it("collapses vary parameters when loading playground state", () => {
    const config = setParameter({}, "steps", { mode: "vary", values: [20, 30] });
    globalThis.localStorage.setItem(
      PLAYGROUND_STORAGE_KEY,
      JSON.stringify({ config, mode: "sweep", batchCount: 4 }),
    );
    const loaded = loadPlaygroundState();
    assert.equal(loaded.batchCount, 4);
    const steps = (loaded.config.parameters as { steps?: { mode?: string; value?: unknown } }).steps;
    assert.equal(steps?.mode, "fixed");
    assert.equal(steps?.value, 20);
  });

  it("migrates legacy playground experiment mode into experiment draft storage once", () => {
    const config = setParameter({}, "prompt", { mode: "vary", values: ["a", "b"] });
    globalThis.localStorage.setItem(
      PLAYGROUND_STORAGE_KEY,
      JSON.stringify({ config, mode: "sweep", batchCount: 1 }),
    );

    const migrated = loadExperimentDraft();
    const prompt = (migrated.config.parameters as { prompt?: { mode?: string; values?: unknown[] } }).prompt;
    assert.equal(prompt?.mode, "vary");
    assert.deepEqual(prompt?.values, ["a", "b"]);

    globalThis.localStorage.setItem(
      PLAYGROUND_STORAGE_KEY,
      JSON.stringify({ config: {}, mode: "sweep" }),
    );
    const again = loadExperimentDraft();
    const promptAgain = (again.config.parameters as { prompt?: { values?: unknown[] } }).prompt;
    assert.deepEqual(promptAgain?.values, ["a", "b"]);
  });

  it("does not overwrite an existing experiment draft during playground load", () => {
    globalThis.localStorage.setItem(
      EXPERIMENT_DRAFT_KEY,
      JSON.stringify({ config: setParameter({}, "prompt", { mode: "fixed", value: "keep me" }) }),
    );
    globalThis.localStorage.setItem(
      PLAYGROUND_STORAGE_KEY,
      JSON.stringify({
        config: setParameter({}, "prompt", { mode: "vary", values: ["stolen"] }),
        mode: "sweep",
      }),
    );

    loadPlaygroundState();
    const draft = loadExperimentDraft();
    const prompt = (draft.config.parameters as { prompt?: { value?: unknown } }).prompt;
    assert.equal(prompt?.value, "keep me");
  });

  it("clears a stale hidden LoRA stack when loading playground state", () => {
    const config = applyLoraStack(
      setParameter({}, "prompt", { mode: "fixed", value: "portrait <lora:style:0.8>" }),
      [
        { path: "a.safetensors", trigger: "", weight: 0.8 },
        { path: "b.safetensors", trigger: "", weight: 0.4 },
      ],
    );
    globalThis.localStorage.setItem(
      PLAYGROUND_STORAGE_KEY,
      JSON.stringify({ config, batchCount: 1 }),
    );

    const loaded = loadPlaygroundState();
    const parameters = getParameters(loaded.config);
    assert.deepEqual(loaded.config.lora_paths, []);
    assert.deepEqual(parameters.lora_path?.value, { path: null, trigger: "", weight: 1, bypassed: false });
    assert.equal(parameters.lora_weight?.value, 1);
    assert.equal(parameters.prompt?.value, "portrait <lora:style:0.8>");
  });
});

describe("collapseExperimentToFixed", () => {
  it("keeps the first vary value as the fixed value", () => {
    const collapsed = collapseExperimentToFixed(
      setParameter({}, "cfg_scale", { mode: "vary", values: [5, 7.5] }),
    );
    const cfg = (collapsed.parameters as { cfg_scale?: { mode?: string; value?: unknown } }).cfg_scale;
    assert.equal(cfg?.mode, "fixed");
    assert.equal(cfg?.value, 5);
  });
});
