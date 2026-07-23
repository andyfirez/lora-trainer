import { api } from "@/lib/api/client";

export interface GpuInfo {
  cuda_available: boolean;
  device_name: string | null;
  device_count: number;
  vram_gb: number[] | null;
}

export interface ServerInfo {
  host: string;
  port: number;
}

export interface DatabaseInfo {
  path: string;
  echo: boolean;
}

export interface TrainingSystemInfo {
  logs_dir: string;
  cancel_poll_interval_seconds: number;
}

export interface Settings {
  max_concurrent_jobs: number;
  worker_poll_interval_seconds: number;
  server: ServerInfo;
  database: DatabaseInfo;
  training: TrainingSystemInfo;
  config_file: string;
  app_version: string;
  gpu: GpuInfo;
}

export interface SettingsPatch {
  max_concurrent_jobs?: number;
  worker_poll_interval_seconds?: number;
}

export const settingsApi = {
  get: () => api.get<Settings>("/settings/"),
  patch: (data: SettingsPatch) => api.patch<Settings>("/settings/", data),
};
