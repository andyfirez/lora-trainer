import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { getBaseModelPathFormatIssue } from "./validateBaseModelPathFormat.ts";

describe("getBaseModelPathFormatIssue", () => {
  it("requires a non-empty path", () => {
    assert.match(getBaseModelPathFormatIssue("") ?? "", /Select an SDXL/);
  });

  it("accepts safetensors and ckpt paths", () => {
    assert.equal(getBaseModelPathFormatIssue("D:\\models\\sdxl.safetensors"), null);
    assert.equal(getBaseModelPathFormatIssue("checkpoints/model.ckpt"), null);
  });

  it("rejects huggingface-style ids and folders", () => {
    assert.ok(getBaseModelPathFormatIssue("stabilityai/stable-diffusion-xl-base-1.0"));
    assert.ok(getBaseModelPathFormatIssue("C:\\ComfyUI\\models\\sdxl"));
  });
});
