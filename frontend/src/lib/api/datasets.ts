import { api } from "@/lib/api/client";
import type { Dataset, DatasetImages } from "@/types";

export const datasetsApi = {
  list: () => api.get<Dataset[]>("/datasets/"),
  get: (id: number) => api.get<Dataset>(`/datasets/${id}`),
  create: (data: { name: string; image_dir: string; caption_dir?: string; description?: string }) =>
    api.post<Dataset>("/datasets/", data),
  update: (id: number, data: Partial<{ name: string; image_dir: string; caption_dir: string; description: string }>) =>
    api.patch<Dataset>(`/datasets/${id}`, data),
  delete: (id: number) => api.delete(`/datasets/${id}`),
  listImages: (id: number) => api.get<DatasetImages>(`/datasets/${id}/images`),
};
