import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { partitionLoraFolderContents } from "./loraFolderContents.ts";

describe("partitionLoraFolderContents", () => {
  const loras = [
    { id: 1, relative_path: "anime/girl_01", name: "girl_01" },
    { id: 2, relative_path: "flat_lora", name: "flat_lora" },
    { id: 3, relative_path: "", name: "draft-run" },
  ] as const;

  it("shows catalog loras and plain folders separately", () => {
    const entries = [
      { name: "anime", relative_path: "anime", is_dir: true, is_lora_work_dir: false },
      { name: "misc", relative_path: "misc", is_dir: true, is_lora_work_dir: false },
      { name: "flat_lora", relative_path: "flat_lora", is_dir: true, is_lora_work_dir: true },
    ];

    const root = partitionLoraFolderContents({ entries, loras: [...loras], currentPath: "" });
    assert.equal(root.folders.length, 2);
    assert.equal(root.loras.length, 2);
    assert.equal(root.loras.some((lora) => lora.id === 2), true);
    assert.equal(root.loras.some((lora) => lora.id === 3), true);
    assert.equal(root.unmatchedLoraDirs.length, 0);
  });

  it("treats lora work dirs as items instead of navigable folders", () => {
    const entries = [
      { name: "girl_01", relative_path: "anime/girl_01", is_dir: true, is_lora_work_dir: true },
    ];

    const nested = partitionLoraFolderContents({ entries, loras: [...loras], currentPath: "anime" });
    assert.equal(nested.folders.length, 0);
    assert.equal(nested.loras.length, 1);
    assert.equal(nested.loras[0]?.id, 1);
  });

  it("lists unmatched lora work dirs when catalog is stale", () => {
    const entries = [
      { name: "fresh_lora", relative_path: "fresh_lora", is_dir: true, is_lora_work_dir: true },
    ];

    const result = partitionLoraFolderContents({ entries, loras: [], currentPath: "" });
    assert.equal(result.folders.length, 0);
    assert.equal(result.loras.length, 0);
    assert.equal(result.unmatchedLoraDirs.length, 1);
    assert.equal(result.unmatchedLoraDirs[0]?.relative_path, "fresh_lora");
  });
});
