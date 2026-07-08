import { api } from "@/lib/api/client";
import type { ConfigType, CreateJobFromConfigRequest, Job, JobConfig, JobConfigVersion, JobConfigVersionSummary } from "@/types";

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

export interface JobConfigCloneRequest {
  name?: string;
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
  clone: (id: number, body: JobConfigCloneRequest = {}) =>
    api.post<JobConfig>(`/configs/${id}/clone`, body),
  createJob: (configId: number, body: CreateJobFromConfigRequest = {}) =>
    api.post<Job>(`/configs/${configId}/jobs`, body),
  listVersions: (id: number) => api.get<JobConfigVersionSummary[]>(`/configs/${id}/versions`),
  getVersion: (id: number, version: number) =>
    api.get<JobConfigVersion>(`/configs/${id}/versions/${version}`),
};
