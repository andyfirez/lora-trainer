import { api } from "@/lib/api/client";
import { createRunnableApi } from "@/lib/api/runnableApi";
import type {
  CreateLoraRequest,
  LossBatchResponse,
  LossResponse,
  LoraResponse,
  ReproduceLoraRequest,
} from "@/types";

const runnableApi = createRunnableApi<LoraResponse>("/loras", { cancelWithCheckpoint: true });

export const lorasApi = {
  ...runnableApi,
  create: (body: CreateLoraRequest) => api.post<LoraResponse>("/loras/", body),
  resume: (id: number) => api.post<LoraResponse>(`/loras/${id}/resume`),
  getLoss: (id: number, params: { key?: string; limit?: number; since_step?: number; stride?: number } = {}) => {
    const search = new URLSearchParams();
    if (params.key) search.set("key", params.key);
    if (params.limit != null) search.set("limit", String(params.limit));
    if (params.since_step != null) search.set("since_step", String(params.since_step));
    if (params.stride != null) search.set("stride", String(params.stride));
    const qs = search.toString();
    return api.get<LossResponse>(`/loras/${id}/loss${qs ? `?${qs}` : ""}`);
  },
  getLossBatch: (
    id: number,
    params: { keys?: string[]; sinceSteps?: Array<number | null>; limit?: number; stride?: number } = {},
  ) => {
    const search = new URLSearchParams();
    if (params.keys?.length) {
      search.set("keys", params.keys.join(","));
    } else {
      search.set("keys", "*");
    }
    if (params.sinceSteps?.length) {
      search.set(
        "since_steps",
        params.sinceSteps.map((step) => (step == null ? "" : String(step))).join(","),
      );
    }
    if (params.limit != null) search.set("limit", String(params.limit));
    if (params.stride != null) search.set("stride", String(params.stride));
    const qs = search.toString();
    return api.get<LossBatchResponse>(`/loras/${id}/loss/batch${qs ? `?${qs}` : ""}`);
  },
  reproduce: (id: number, body: ReproduceLoraRequest = {}) =>
    api.post<LoraResponse>(`/loras/${id}/reproduce`, body),
};
