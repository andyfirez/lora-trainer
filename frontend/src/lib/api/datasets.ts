import { BASE_URL, api } from "@/lib/api/client";
import type {
  AutotagRequest,
  AutotagStatusResponse,
  BulkTagResult,
  CropMeta,
  Dataset,
  DatasetImages,
  DatasetItem,
  DatasetItems,
  DuplicatesInfo,
  PreprocessStatus,
  RemoveDuplicatesResult,
  TagStat,
  TagStats,
} from "@/types";

function withCaptionExtension(path: string, captionExtension: string): string {
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}caption_extension=${encodeURIComponent(captionExtension)}`;
}

export function datasetImageUrl(
  datasetId: number,
  filename: string,
  options: { prepared?: boolean; width?: number; cacheKey?: string } = {},
): string {
  const { prepared = false, width = 256, cacheKey } = options;
  const suffix = prepared ? "/prepared" : "";
  const base = `${BASE_URL}/datasets/${datasetId}/images/${encodeURIComponent(filename)}${suffix}?w=${width}`;
  return cacheKey ? `${base}&v=${encodeURIComponent(cacheKey)}` : base;
}

export function datasetPreparedImageUrl(
  datasetId: number,
  filename: string,
  width = 256,
  cacheKey?: string,
): string {
  return datasetImageUrl(datasetId, filename, { prepared: true, width, cacheKey });
}

export function datasetCropPreviewUrl(datasetId: number, filename: string): string {
  return `${BASE_URL}/datasets/${datasetId}/images/${encodeURIComponent(filename)}/crop-preview`;
}

export const datasetsApi = {
  list: () => api.get<Dataset[]>("/datasets/"),
  get: (id: number) => api.get<Dataset>(`/datasets/${id}`),
  create: (data: { name: string; relative_path: string; description?: string }) =>
    api.post<Dataset>("/datasets/", data),
  import: (data: { name: string; source_dir: string; relative_path: string; description?: string }) =>
    api.post<Dataset>("/datasets/import", data),
  update: (
    id: number,
    data: Partial<{
      name: string;
      relative_path: string;
      description: string;
      target_resolution: number | null;
      enable_bucket: boolean;
      bucket_reso_steps: number;
      min_bucket_reso: number;
      max_bucket_reso: number;
      bucket_no_upscale: boolean;
    }>
  ) => api.patch<Dataset>(`/datasets/${id}`, data),
  delete: (id: number) => api.delete(`/datasets/${id}`),
  listImages: (id: number) => api.get<DatasetImages>(`/datasets/${id}/images`),
  listItems: (id: number, captionExtension = ".txt") =>
    api.get<DatasetItems>(withCaptionExtension(`/datasets/${id}/items`, captionExtension)),
  getPreprocessStatus: (id: number) => api.get<PreprocessStatus>(`/datasets/${id}/preprocess/status`),
  getDuplicates: (id: number) => api.get<DuplicatesInfo>(`/datasets/${id}/duplicates`),
  removeDuplicates: (id: number, captionExtension = ".txt") =>
    api.post<RemoveDuplicatesResult>(withCaptionExtension(`/datasets/${id}/duplicates/remove`, captionExtension), {}),
  deleteImage: (id: number, filename: string, captionExtension = ".txt") =>
    api.delete(
      withCaptionExtension(
        `/datasets/${id}/images/${encodeURIComponent(filename)}`,
        captionExtension,
      ),
    ),
  getCropMeta: (id: number, filename: string) =>
    api.get<CropMeta>(`/datasets/${id}/images/${encodeURIComponent(filename)}/crop-meta`),
  saveCrop: (id: number, filename: string, crop_center_x: number, crop_center_y: number) =>
    api.put<CropMeta>(`/datasets/${id}/images/${encodeURIComponent(filename)}/crop`, {
      crop_center_x,
      crop_center_y,
    }),
  bakeImage: (id: number, filename: string) =>
    api.post<CropMeta>(`/datasets/${id}/images/${encodeURIComponent(filename)}/bake`, {}),
  bakeAll: (id: number, filenames?: string[]) =>
    api.post<{ baked_count: number; preprocess_ready: boolean }>(`/datasets/${id}/preprocess/bake`, {
      filenames,
    }),
  getCaption: (id: number, filename: string, captionExtension = ".txt") =>
    api.get<{ filename: string; tags: string[] }>(
      withCaptionExtension(`/datasets/${id}/captions/${encodeURIComponent(filename)}`, captionExtension),
    ),
  updateCaption: (id: number, filename: string, tags: string[], captionExtension = ".txt") =>
    api.put<{ filename: string; tags: string[] }>(
      withCaptionExtension(`/datasets/${id}/captions/${encodeURIComponent(filename)}`, captionExtension),
      { tags },
    ),
  getTagStats: (id: number, captionExtension = ".txt") =>
    api.get<TagStats>(withCaptionExtension(`/datasets/${id}/tags/stats`, captionExtension)),
  bulkAddTag: (id: number, tag: string, filenames?: string[], captionExtension = ".txt") =>
    api.post<BulkTagResult>(`/datasets/${id}/tags/bulk-add`, {
      tag,
      filenames,
      caption_extension: captionExtension,
    }),
  bulkRemoveTag: (id: number, tag: string, filenames?: string[], captionExtension = ".txt") =>
    api.post<BulkTagResult>(`/datasets/${id}/tags/bulk-remove`, {
      tag,
      filenames,
      caption_extension: captionExtension,
    }),
  autotag: (id: number, body: AutotagRequest = {}) =>
    api.post<AutotagStatusResponse>(`/datasets/${id}/autotag`, body),
  getAutotagStatus: (id: number) => api.get<AutotagStatusResponse>(`/datasets/${id}/autotag/status`),
};

export type { DatasetItem, TagStat };
