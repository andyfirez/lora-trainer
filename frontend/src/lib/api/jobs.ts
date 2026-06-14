import { api } from "@/lib/api/client";
import type { Job, SamplingRun } from "@/types";

export interface JobLogs {
  lines: string[];
}

export interface LossPoint {
  step: number;
  wall_time?: number | null;
  value: number | null;
}

export interface JobLossResponse {
  key: string;
  keys: string[];
  points: LossPoint[];
}

export const jobsApi = {
  list: () => api.get<Job[]>("/jobs/"),
  get: (id: number) => api.get<Job>(`/jobs/${id}`),
  create: (name: string, config_yaml: string) => api.post<Job>("/jobs/", { name, config_yaml }),
  update: (id: number, data: { name?: string; config_yaml?: string }) => api.patch<Job>(`/jobs/${id}`, data),
  delete: (id: number) => api.delete(`/jobs/${id}`),
  enqueue: (id: number) => api.post<Job>(`/jobs/${id}/enqueue`),
  resume: (id: number) => api.post<Job>(`/jobs/${id}/resume`),
  cancel: (id: number, saveCheckpoint = false) =>
    api.post<Job>(`/jobs/${id}/cancel${saveCheckpoint ? "?save_checkpoint=true" : ""}`),
  getLogs: (id: number, tail = 500) => api.get<JobLogs>(`/jobs/${id}/logs?tail=${tail}`),
  getLoss: (id: number, params: { key?: string; limit?: number; since_step?: number; stride?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.key) search.set("key", params.key);
    if (params.limit != null) search.set("limit", String(params.limit));
    if (params.since_step != null) search.set("since_step", String(params.since_step));
    if (params.stride != null) search.set("stride", String(params.stride));
    const qs = search.toString();
    return api.get<JobLossResponse>(`/jobs/${id}/loss${qs ? `?${qs}` : ""}`);
  },
  listSamplingRuns: (id: number) => api.get<SamplingRun[]>(`/jobs/${id}/sampling-runs`),
  createSamplingRun: (id: number, data: { lora_paths: string[]; name?: string; enqueue?: boolean }) =>
    api.post<SamplingRun>(`/jobs/${id}/sample`, data),
};