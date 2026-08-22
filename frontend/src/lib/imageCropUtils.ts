import type { MediaSize } from "react-easy-crop";
import type { CropMeta } from "@/types";

export interface Area {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface CropSelectionRect {
  selection_x: number;
  selection_y: number;
  selection_width: number;
  selection_height: number;
}

export function centerToArea(meta: CropMeta, cropWidth: number, cropHeight: number): Area {
  const halfW = cropWidth / 2;
  const halfH = cropHeight / 2;
  const x = Math.max(0, Math.min(meta.crop_center_x * meta.fitted_width - halfW, meta.fitted_width - cropWidth));
  const y = Math.max(0, Math.min(meta.crop_center_y * meta.fitted_height - halfH, meta.fitted_height - cropHeight));
  return { x, y, width: cropWidth, height: cropHeight };
}

export function areaToCenter(area: Area, meta: CropMeta): { x: number; y: number } {
  const halfW = area.width / 2;
  const halfH = area.height / 2;
  const cx = Math.max(halfW, Math.min(area.x + halfW, meta.fitted_width - halfW));
  const cy = Math.max(halfH, Math.min(area.y + halfH, meta.fitted_height - halfH));
  return {
    x: cx / meta.fitted_width,
    y: cy / meta.fitted_height,
  };
}

export function normalizeSelectionForSave(area: Area): CropSelectionRect {
  return {
    selection_x: Math.round(area.x),
    selection_y: Math.round(area.y),
    selection_width: Math.max(1, Math.round(area.width)),
    selection_height: Math.max(1, Math.round(area.height)),
  };
}

export function selectionToInitialArea(meta: CropMeta): Area | null {
  if (
    meta.selection_x == null ||
    meta.selection_y == null ||
    meta.selection_width == null ||
    meta.selection_height == null
  ) {
    return null;
  }
  return {
    x: meta.selection_x,
    y: meta.selection_y,
    width: meta.selection_width,
    height: meta.selection_height,
  };
}

export function cropDimensions(
  meta: CropMeta,
  targetResolution: number,
): { width: number; height: number } {
  if (meta.enable_bucket && meta.bucket_width && meta.bucket_height) {
    return { width: meta.bucket_width, height: meta.bucket_height };
  }
  return { width: targetResolution, height: targetResolution };
}

export function computeDisplayCropSize(
  mediaSize: MediaSize,
  cropWidth: number,
  cropHeight: number,
): { width: number; height: number } {
  const scale = mediaSize.width / mediaSize.naturalWidth;
  const width = Math.round(cropWidth * scale);
  const height = Math.round(cropHeight * scale);
  return {
    width: Math.min(width, Math.floor(mediaSize.width)),
    height: Math.min(height, Math.floor(mediaSize.height)),
  };
}

export function previewCanvasSize(cropWidth: number, cropHeight: number, previewMax = 160): {
  width: number;
  height: number;
} {
  const scale = previewMax / Math.max(cropWidth, cropHeight);
  return {
    width: Math.round(cropWidth * scale),
    height: Math.round(cropHeight * scale),
  };
}
