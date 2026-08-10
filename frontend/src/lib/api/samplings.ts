import { api } from "@/lib/api/client";
import type { RunnableSamplesResponse, SamplingResponse, SweepManifestResponse } from "@/types";
import type { LogsResponse } from "@/lib/api/loras";

export interface CreateSamplingRequest {
  name: string;
  config_yaml: string;
  lora_paths?: string[];
}

export const samplingsApi = {
  list: () => api.get<SamplingResponse[]>("/samplings/"),
  create: (body: CreateSamplingRequest) => api.post<SamplingResponse>("/samplings/", body),
  get: (id: number) => api.get<SamplingResponse>(`/samplings/${id}`),
  enqueue: (id: number) => api.post<SamplingResponse>(`/samplings/${id}/enqueue`),
  cancel: (id: number) => api.post<SamplingResponse>(`/samplings/${id}/cancel`),
  getLogs: (id: number, tail = 500) => api.get<LogsResponse>(`/samplings/${id}/logs?tail=${tail}`),
  getSamples: (id: number) => api.get<RunnableSamplesResponse>(`/samplings/${id}/samples`),
  getSweepManifest: (id: number) => api.get<SweepManifestResponse | null>(`/samplings/${id}/sweep-manifest`),
};
