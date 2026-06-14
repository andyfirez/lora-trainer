import { api } from "@/lib/api/client";
import type { JobLogs } from "@/lib/api/jobs";
import type { SamplingRun, SamplingRunSamplesResponse } from "@/types";

export const samplingRunsApi = {
  list: (sourceJobId?: number) =>
    api.get<SamplingRun[]>(
      sourceJobId != null ? `/sampling-runs/?source_job_id=${sourceJobId}` : "/sampling-runs/",
    ),
  get: (id: number) => api.get<SamplingRun>(`/sampling-runs/${id}`),
  cancel: (id: number) => api.post<SamplingRun>(`/sampling-runs/${id}/cancel`),
  enqueue: (id: number) => api.post<SamplingRun>(`/sampling-runs/${id}/enqueue`),
  getLogs: (id: number, tail = 500) => api.get<JobLogs>(`/sampling-runs/${id}/logs?tail=${tail}`),
  getSamples: (id: number) => api.get<SamplingRunSamplesResponse>(`/sampling-runs/${id}/samples`),
};
