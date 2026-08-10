import { api, BASE_URL } from "@/lib/api/client";
import type { LoraResponse, RunnableSamplesResponse } from "@/types";

export interface LogsResponse {
  lines: string[];
}

export interface LossPoint {
  step: number;
  wall_time?: number | null;
  value: number | null;
}

export interface LossResponse {
  key: string;
  keys: string[];
  points: LossPoint[];
}

export interface CreateLoraRequest {
  name: string;
  config_yaml: string;
}

export interface ReproduceLoraRequest {
  name?: string;
  enqueue?: boolean;
}

export const lorasApi = {
  list: () => api.get<LoraResponse[]>("/loras/"),
  create: (body: CreateLoraRequest) => api.post<LoraResponse>("/loras/", body),
  get: (id: number) => api.get<LoraResponse>(`/loras/${id}`),
  enqueue: (id: number) => api.post<LoraResponse>(`/loras/${id}/enqueue`),
  resume: (id: number) => api.post<LoraResponse>(`/loras/${id}/resume`),
  cancel: (id: number, saveCheckpoint = false) =>
    api.post<LoraResponse>(`/loras/${id}/cancel`, { save_checkpoint: saveCheckpoint }),
  getLogs: (id: number, tail = 500) => api.get<LogsResponse>(`/loras/${id}/logs?tail=${tail}`),
  getLoss: (id: number, params: { key?: string; limit?: number; since_step?: number; stride?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.key) search.set("key", params.key);
    if (params.limit != null) search.set("limit", String(params.limit));
    if (params.since_step != null) search.set("since_step", String(params.since_step));
    if (params.stride != null) search.set("stride", String(params.stride));
    const qs = search.toString();
    return api.get<LossResponse>(`/loras/${id}/loss${qs ? `?${qs}` : ""}`);
  },
  getSamples: (id: number) => api.get<RunnableSamplesResponse>(`/loras/${id}/samples`),
  downloadWeightsUrl: (id: number) => `${BASE_URL}/loras/${id}/weights`,
  reproduce: (id: number, body: ReproduceLoraRequest = {}) =>
    api.post<LoraResponse>(`/loras/${id}/reproduce`, body),
};
