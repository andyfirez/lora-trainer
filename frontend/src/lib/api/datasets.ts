import { BASE_URL, api } from "@/lib/api/client";
import type {
  AutotagRequest,
  AutotagResponse,
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

export function datasetImageUrl(datasetId: number, filename: string, width = 256, cacheKey?: string): string {
  const base = `${BASE_URL}/datasets/${datasetId}/images/${encodeURIComponent(filename)}?w=${width}`;
  return cacheKey ? `${base}&v=${encodeURIComponent(cacheKey)}` : base;
}

export function datasetPreparedImageUrl(
  datasetId: number,
  filename: string,
  width = 256,
  cacheKey?: string
): string {
  const base = `${BASE_URL}/datasets/${datasetId}/images/${encodeURIComponent(filename)}/prepared?w=${width}`;
  return cacheKey ? `${base}&v=${encodeURIComponent(cacheKey)}` : base;
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
    api.get<DatasetItems>(`/datasets/${id}/items?caption_extension=${encodeURIComponent(captionExtension)}`),
  getPreprocessStatus: (id: number) => api.get<PreprocessStatus>(`/datasets/${id}/preprocess/status`),
  getDuplicates: (id: number) => api.get<DuplicatesInfo>(`/datasets/${id}/duplicates`),
  removeDuplicates: (id: number, captionExtension = ".txt") =>
    api.post<RemoveDuplicatesResult>(
      `/datasets/${id}/duplicates/remove?caption_extension=${encodeURIComponent(captionExtension)}`,
      {}
    ),
  deleteImage: (id: number, filename: string, captionExtension = ".txt") =>
    api.delete(
      `/datasets/${id}/images/${encodeURIComponent(filename)}?caption_extension=${encodeURIComponent(captionExtension)}`
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
      `/datasets/${id}/captions/${encodeURIComponent(filename)}?caption_extension=${encodeURIComponent(captionExtension)}`
    ),
  updateCaption: (id: number, filename: string, tags: string[], captionExtension = ".txt") =>
    api.put<{ filename: string; tags: string[] }>(
      `/datasets/${id}/captions/${encodeURIComponent(filename)}?caption_extension=${encodeURIComponent(captionExtension)}`,
      { tags }
    ),
  getTagStats: (id: number, captionExtension = ".txt") =>
    api.get<TagStats>(`/datasets/${id}/tags/stats?caption_extension=${encodeURIComponent(captionExtension)}`),
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
    api.post<AutotagResponse>(`/datasets/${id}/autotag`, body),
};

export type { DatasetItem, TagStat };
