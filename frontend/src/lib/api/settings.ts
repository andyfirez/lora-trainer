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

export interface StorageInfo {
  datasets_root: string;
  base_models_root: string;
  lora_root: string;
}

export interface GpuDefaultsInfo {
  tf32: boolean;
  attention_mechanism: "default" | "sdpa" | "xformers";
  mixed_precision: "float32" | "float16" | "bfloat16";
  vae_dtype: "auto" | "float32" | "float16" | "bfloat16";
  sample_vae_tiling: boolean;
}

export interface Settings {
  max_concurrent_jobs: number;
  worker_poll_interval_seconds: number;
  server: ServerInfo;
  database: DatabaseInfo;
  storage: StorageInfo;
  training: TrainingSystemInfo;
  gpu_defaults: GpuDefaultsInfo;
  config_file: string;
  app_version: string;
  gpu: GpuInfo;
}

export interface SettingsPatch {
  max_concurrent_jobs?: number;
  worker_poll_interval_seconds?: number;
  datasets_root?: string;
  base_models_root?: string;
  lora_root?: string;
  tf32?: boolean;
  attention_mechanism?: GpuDefaultsInfo["attention_mechanism"];
  mixed_precision?: GpuDefaultsInfo["mixed_precision"];
  vae_dtype?: GpuDefaultsInfo["vae_dtype"];
  sample_vae_tiling?: boolean;
}

export const settingsApi = {
  get: () => api.get<Settings>("/settings/"),
  patch: (data: SettingsPatch) => api.patch<Settings>("/settings/", data),
};
