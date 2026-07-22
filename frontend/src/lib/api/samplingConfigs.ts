import { api } from "@/lib/api/client";
import type { CreateJobFromConfigRequest, Job, JobConfig } from "@/types";

export interface SamplingConfigCreate {
  name: string;
  config_yaml: string;
  description?: string | null;
}

export interface SamplingConfigUpdate {
  name?: string;
  config_yaml?: string;
  description?: string | null;
}

export interface SamplingConfigCloneRequest {
  name?: string;
  description?: string | null;
}

export const samplingConfigsApi = {
  list: () => api.get<JobConfig[]>("/sampling-configs/"),
  get: (id: number) => api.get<JobConfig>(`/sampling-configs/${id}`),
  create: (body: SamplingConfigCreate) =>
    api.post<JobConfig>("/sampling-configs/", body),
  update: (id: number, body: SamplingConfigUpdate) =>
    api.patch<JobConfig>(`/sampling-configs/${id}`, body),
  delete: (id: number) => api.delete(`/sampling-configs/${id}`),
  clone: (id: number, body: SamplingConfigCloneRequest = {}) =>
    api.post<JobConfig>(`/sampling-configs/${id}/clone`, body),
  createJob: (configId: number, body: CreateJobFromConfigRequest = {}) =>
    api.post<Job>(`/sampling-configs/${configId}/jobs`, body),
};
