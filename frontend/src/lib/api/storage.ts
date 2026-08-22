import { api } from "@/lib/api/client";

export type StorageKind = "datasets" | "base_models" | "lora";

export interface StorageEntry {
  name: string;
  relative_path: string;
  is_dir: boolean;
  is_lora_work_dir?: boolean;
}

export interface StorageBrowseResponse {
  kind: StorageKind;
  root: string;
  relative_path: string;
  entries: StorageEntry[];
}

export interface BaseModelsListResponse {
  root: string;
  models: StorageEntry[];
}

export const storageApi = {
  browse: (kind: StorageKind, relativePath = "") =>
    api.get<StorageBrowseResponse>(
      `/storage/browse?kind=${encodeURIComponent(kind)}&relative_path=${encodeURIComponent(relativePath)}`
    ),

  listBaseModels: () => api.get<BaseModelsListResponse>("/storage/base-models"),
};
