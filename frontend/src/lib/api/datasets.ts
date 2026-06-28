import { BASE_URL, api } from "@/lib/api/client";
import type {
  AutotagRequest,
  AutotagResponse,
  BulkTagResult,
  Dataset,
  DatasetImages,
  DatasetItem,
  DatasetItems,
  TagStat,
  TagStats,
} from "@/types";

export function datasetImageUrl(datasetId: number, filename: string, width = 256): string {
  return `${BASE_URL}/datasets/${datasetId}/images/${encodeURIComponent(filename)}?w=${width}`;
}

export const datasetsApi = {
  list: () => api.get<Dataset[]>("/datasets/"),
  get: (id: number) => api.get<Dataset>(`/datasets/${id}`),
  create: (data: { name: string; image_dir: string; description?: string }) =>
    api.post<Dataset>("/datasets/", data),
  update: (id: number, data: Partial<{ name: string; image_dir: string; caption_dir: string; description: string }>) =>
    api.patch<Dataset>(`/datasets/${id}`, data),
  delete: (id: number) => api.delete(`/datasets/${id}`),
  listImages: (id: number) => api.get<DatasetImages>(`/datasets/${id}/images`),
  listItems: (id: number, captionExtension = ".txt") =>
    api.get<DatasetItems>(`/datasets/${id}/items?caption_extension=${encodeURIComponent(captionExtension)}`),
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
