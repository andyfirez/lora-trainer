import { api, BASE_URL } from "@/lib/api/client";
import type { Job, TrainedLora, JobSamplesResponse } from "@/types";

export interface ReproduceTrainedLoraRequest {
  name?: string;
  enqueue?: boolean;
}

export const lorasApi = {
  list: () => api.get<TrainedLora[]>("/loras/"),
  get: (id: number) => api.get<TrainedLora>(`/loras/${id}`),
  getSamples: (id: number) => api.get<JobSamplesResponse>(`/loras/${id}/samples`),
  downloadWeightsUrl: (id: number) => `${BASE_URL}/loras/${id}/weights`,
  reproduce: (id: number, body: ReproduceTrainedLoraRequest = {}) =>
    api.post<Job>(`/loras/${id}/reproduce`, body),
};
