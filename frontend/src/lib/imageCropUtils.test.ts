import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  areaToCenter,
  centerToArea,
  normalizeSelectionForSave,
  selectionToInitialArea,
} from "./imageCropUtils.ts";
import type { CropMeta } from "../types/index.ts";

const baseMeta: CropMeta = {
  crop_center_x: 0.5,
  crop_center_y: 0.5,
  fitted_width: 1024,
  fitted_height: 1536,
  source_width: 800,
  source_height: 1200,
  state: "ready",
};

describe("imageCropUtils", () => {
  it("round-trips center through area", () => {
    const area = centerToArea(baseMeta, 512, 512);
    const center = areaToCenter(area, baseMeta);
    assert.equal(center.x, 0.5);
    assert.equal(center.y, 0.5);
  });

  it("preserves zoomed selection dimensions in save payload", () => {
    const selection = normalizeSelectionForSave({ x: 120, y: 80, width: 256, height: 256 });
    assert.equal(selection.selection_width, 256);
    assert.equal(selection.selection_height, 256);
    assert.equal(selection.selection_x, 120);
    assert.equal(selection.selection_y, 80);
  });

  it("restores stored selection for cropper init", () => {
    const meta: CropMeta = {
      ...baseMeta,
      selection_x: 10,
      selection_y: 20,
      selection_width: 300,
      selection_height: 280,
    };
    const area = selectionToInitialArea(meta);
    assert.deepEqual(area, { x: 10, y: 20, width: 300, height: 280 });
  });
});
