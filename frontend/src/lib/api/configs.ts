import { api } from "@/lib/api/client";
import type { ConfigType, CreateJobFromConfigRequest, Job, JobConfig } from "@/types";

export interface JobConfigCreate {
  name: string;
  config_type: ConfigType;
  config_yaml: string;
  description?: string | null;
}

export interface JobConfigUpdate {
  name?: string;
  config_yaml?: string;
  description?: string | null;
}

export const configsApi = {
  list: (configType?: ConfigType) =>
    api.get<JobConfig[]>(
      configType != null ? `/configs/?config_type=${configType}` : "/configs/",
    ),
  get: (id: number) => api.get<JobConfig>(`/configs/${id}`),
  create: (body: JobConfigCreate) => api.post<JobConfig>("/configs/", body),
  update: (id: number, body: JobConfigUpdate) => api.patch<JobConfig>(`/configs/${id}`, body),
  delete: (id: number) => api.delete(`/configs/${id}`),
  createJob: (configId: number, body: CreateJobFromConfigRequest = {}) =>
    api.post<Job>(`/configs/${configId}/jobs`, body),
};
