import { api } from "@/lib/api/client";
import { createRunnableApi } from "@/lib/api/runnableApi";
import type { CreateSamplingRequest, SamplingResponse, SweepManifestResponse } from "@/types";

const runnableApi = createRunnableApi<SamplingResponse>("/samplings");

export const samplingsApi = {
  ...runnableApi,
  create: (body: CreateSamplingRequest) => api.post<SamplingResponse>("/samplings/", body),
  getSweepManifest: (id: number) => api.get<SweepManifestResponse | null>(`/samplings/${id}/sweep-manifest`),
};
