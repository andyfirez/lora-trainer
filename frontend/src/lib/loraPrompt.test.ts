import assert from "node:assert/strict";
import test from "node:test";
import { appendLoraTag } from "./loraPrompt.ts";

test("appendLoraTag appends tag to existing prompt", () => {
  assert.equal(appendLoraTag("a cat", "style", 0.8), "a cat <lora:style:0.8>");
});

test("appendLoraTag returns only tag for empty prompt", () => {
  assert.equal(appendLoraTag("", "style", 1), "<lora:style:1>");
});
