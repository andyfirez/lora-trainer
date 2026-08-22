import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  applyLoraStack,
  collapseStackToSingleLora,
  loraParameterFromStack,
  parseLoraEntries,
  parseLoraEntry,
} from "./sweepUtils.ts";

describe("lora stack helpers", () => {
  it("parses weight from an entry object", () => {
    const entry = parseLoraEntry({ path: "a.safetensors", trigger: "ohwx", weight: 0.55 });
    assert.equal(entry.path, "a.safetensors");
    assert.equal(entry.weight, 0.55);
    assert.equal(entry.trigger, "ohwx");
  });

  it("parses a fixed list as a stack", () => {
    const entries = parseLoraEntries([
      { path: "a.safetensors", weight: 0.8 },
      { path: "b.safetensors", weight: 0.4 },
    ]);
    assert.equal(entries.length, 2);
    assert.equal(entries[1]?.weight, 0.4);
  });

  it("stores two file LoRAs as a fixed list", () => {
    const param = loraParameterFromStack([
      { path: "a.safetensors", trigger: "", weight: 0.8 },
      { path: "b.safetensors", trigger: "", weight: 0.4 },
    ]);
    assert.equal(param.mode, "fixed");
    assert.ok(Array.isArray(param.value));
    assert.equal((param.value as { path: string }[]).length, 2);
  });

  it("syncs first stack weight onto lora_weight", () => {
    const next = applyLoraStack({}, [
      { path: "a.safetensors", trigger: "", weight: 0.7 },
      { path: "b.safetensors", trigger: "", weight: 0.3 },
    ]);
    const parameters = next.parameters as {
      lora_weight?: { value?: unknown };
      lora_path?: { mode?: string; value?: unknown };
    };
    assert.equal(parameters.lora_weight?.value, 0.7);
    assert.equal(parameters.lora_path?.mode, "fixed");
    assert.ok(Array.isArray(parameters.lora_path?.value));
  });

  it("collapses a stack to the first LoRA when leaving normal mode", () => {
    const stacked = applyLoraStack({}, [
      { path: "a.safetensors", trigger: "ohwx", weight: 0.8 },
      { path: "b.safetensors", trigger: "sks", weight: 0.4 },
    ]);
    const collapsed = collapseStackToSingleLora(stacked);
    const parameters = collapsed.parameters as {
      lora_path?: { value?: { path?: string } };
    };
    assert.equal(parameters.lora_path?.value?.path, "a.safetensors");
  });
});
