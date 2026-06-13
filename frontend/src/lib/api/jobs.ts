import { api } from "@/lib/api/client";
import type { Job } from "@/types";

export interface JobLogs {
  lines: string[];
}

export const jobsApi = {
  list: () => api.get<Job[]>("/jobs/"),
  get: (id: number) => api.get<Job>(`/jobs/${id}`),
  create: (name: string, config_yaml: string) => api.post<Job>("/jobs/", { name, config_yaml }),
  update: (id: number, data: { name?: string; config_yaml?: string }) => api.patch<Job>(`/jobs/${id}`, data),
  delete: (id: number) => api.delete(`/jobs/${id}`),
  enqueue: (id: number) => api.post<Job>(`/jobs/${id}/enqueue`),
  cancel: (id: number) => api.post<Job>(`/jobs/${id}/cancel`),
  getLogs: (id: number, tail = 500) => api.get<JobLogs>(`/jobs/${id}/logs?tail=${tail}`),
};