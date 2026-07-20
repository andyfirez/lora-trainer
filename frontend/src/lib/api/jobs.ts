import { api } from "@/lib/api/client";
import type { Job, JobSamplesResponse, JobType, SweepManifestResponse } from "@/types";

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

export interface JobListParams {
  job_type?: JobType;
  source_job_id?: number;
}

export const jobsApi = {
  list: (params: JobListParams = {}) => {
    const search = new URLSearchParams();
    if (params.job_type) search.set("job_type", params.job_type);
    if (params.source_job_id != null) search.set("source_job_id", String(params.source_job_id));
    const qs = search.toString();
    return api.get<Job[]>(`/jobs/${qs ? `?${qs}` : ""}`);
  },
  get: (id: number) => api.get<Job>(`/jobs/${id}`),
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
  getSamples: (id: number) => api.get<JobSamplesResponse>(`/jobs/${id}/samples`),
  getSweepManifest: (id: number) => api.get<SweepManifestResponse>(`/jobs/${id}/sweep-manifest`),
};
