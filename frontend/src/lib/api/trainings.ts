import { api } from "@/lib/api/client";
import type { CreateJobFromConfigRequest, Job, JobConfig } from "@/types";

export interface TrainingConfigCreate {
  name: string;
  config_yaml: string;
  description?: string | null;
}

export interface TrainingConfigUpdate {
  name?: string;
  config_yaml?: string;
  description?: string | null;
}

export interface TrainingConfigCloneRequest {
  name?: string;
  description?: string | null;
}

export const trainingsApi = {
  list: () => api.get<JobConfig[]>("/trainings/"),
  get: (id: number) => api.get<JobConfig>(`/trainings/${id}`),
  create: (body: TrainingConfigCreate) => api.post<JobConfig>("/trainings/", body),
  update: (id: number, body: TrainingConfigUpdate) =>
    api.patch<JobConfig>(`/trainings/${id}`, body),
  delete: (id: number) => api.delete(`/trainings/${id}`),
  clone: (id: number, body: TrainingConfigCloneRequest = {}) =>
    api.post<JobConfig>(`/trainings/${id}/clone`, body),
  createJob: (configId: number, body: CreateJobFromConfigRequest = {}) =>
    api.post<Job>(`/trainings/${configId}/jobs`, body),
};
