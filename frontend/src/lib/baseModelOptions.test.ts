import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { buildBaseModelSelectOptions } from "./baseModelOptions.ts";

describe("buildBaseModelSelectOptions", () => {
  it("maps discovered models to select options", () => {
    const options = buildBaseModelSelectOptions([
      { relative_path: "sdxl-base", is_dir: true },
      { relative_path: "legacy.safetensors", is_dir: false },
    ]);
    assert.deepEqual(options, [
      { value: "sdxl-base", label: "sdxl-base" },
      { value: "legacy.safetensors", label: "legacy.safetensors" },
    ]);
  });

  it("keeps current config value when it is missing from the folder", () => {
    const options = buildBaseModelSelectOptions(
      [{ relative_path: "sdxl-base", is_dir: true }],
      ["stabilityai/stable-diffusion-xl-base-1.0"],
    );
    assert.equal(options[0]?.value, "stabilityai/stable-diffusion-xl-base-1.0");
    assert.match(String(options[0]?.label), /not in base models folder/);
  });
});
