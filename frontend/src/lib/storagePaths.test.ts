import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  childName,
  isDirectChild,
  joinRelativePath,
  normalizeRelativePath,
  parentPath,
  partitionFolderContents,
} from "./storagePaths.ts";

describe("normalizeRelativePath", () => {
  it("trims slashes and normalizes separators", () => {
    assert.equal(normalizeRelativePath(" anime/girl_01/ "), "anime/girl_01");
    assert.equal(normalizeRelativePath("anime\\girl_01"), "anime/girl_01");
    assert.equal(normalizeRelativePath("/"), "");
  });
});

describe("parentPath", () => {
  it("returns parent folder path", () => {
    assert.equal(parentPath("anime/girl_01"), "anime");
    assert.equal(parentPath("anime"), "");
    assert.equal(parentPath(""), "");
  });
});

describe("childName", () => {
  it("returns the last path segment", () => {
    assert.equal(childName("anime/girl_01"), "girl_01");
    assert.equal(childName("anime"), "anime");
    assert.equal(childName(""), "");
  });
});

describe("joinRelativePath", () => {
  it("joins normalized path segments", () => {
    assert.equal(joinRelativePath("", "girl_01"), "girl_01");
    assert.equal(joinRelativePath("anime", "girl_01"), "anime/girl_01");
    assert.equal(joinRelativePath("anime/", "/girl_01/"), "anime/girl_01");
  });
});

describe("isDirectChild", () => {
  it("detects direct children at root and nested levels", () => {
    assert.equal(isDirectChild("girl_01", ""), true);
    assert.equal(isDirectChild("anime/girl_01", ""), false);
    assert.equal(isDirectChild("anime/girl_01", "anime"), true);
    assert.equal(isDirectChild("anime/girl_01/extra", "anime"), false);
    assert.equal(isDirectChild("styles/anime/girl_01", "styles"), false);
  });
});

describe("partitionFolderContents", () => {
  const catalog = [
    { id: 1, relative_path: "anime/girl_01" },
    { id: 2, relative_path: "flat_dataset" },
  ];

  it("splits browse entries into navigable folders and catalog items", () => {
    const entries = [
      { name: "anime", relative_path: "anime", is_dir: true },
      { name: "portraits", relative_path: "portraits", is_dir: true },
      { name: "readme.txt", relative_path: "readme.txt", is_dir: false },
    ];

    const root = partitionFolderContents({ entries, catalogItems: catalog, currentPath: "" });
    assert.equal(root.folders.length, 2);
    assert.equal(root.folders[0]?.name, "anime");
    assert.equal(root.folders[1]?.name, "portraits");
    assert.equal(root.items.length, 1);
    assert.equal(root.items[0]?.id, 2);

    const anime = partitionFolderContents({ entries: [], catalogItems: catalog, currentPath: "anime" });
    assert.equal(anime.folders.length, 0);
    assert.equal(anime.items.length, 1);
    assert.equal(anime.items[0]?.id, 1);
  });

  it("treats catalog leaf directories as items, not folders", () => {
    const entries = [{ name: "girl_01", relative_path: "anime/girl_01", is_dir: true }];
    const result = partitionFolderContents({ entries, catalogItems: catalog, currentPath: "anime" });
    assert.equal(result.folders.length, 0);
    assert.equal(result.items.length, 1);
  });
});
