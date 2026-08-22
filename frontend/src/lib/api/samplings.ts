import { api, BASE_URL } from "@/lib/api/client";
import { createRunnableApi } from "@/lib/api/runnableApi";
import type { CreateSamplingRequest, SamplingResponse, SweepManifestResponse } from "@/types";

const runnableApi = createRunnableApi<SamplingResponse>("/samplings");

export const samplingsApi = {
  ...runnableApi,
  create: (body: CreateSamplingRequest) => api.post<SamplingResponse>("/samplings/", body),
  getSweepManifest: (id: number) => api.get<SweepManifestResponse | null>(`/samplings/${id}/sweep-manifest`),
  livePreviewUrl: (id: number, cacheBust: number) =>
    `${BASE_URL}/samplings/${id}/live-preview?t=${cacheBust}`,
  generate: async (config: Record<string, unknown>, loraPaths?: string[]) => {
    const created = await api.post<SamplingResponse>("/samplings/", {
      name: "Playground",
      config,
      lora_paths: loraPaths,
    } satisfies CreateSamplingRequest);
    return runnableApi.enqueue(created.id);
  },
};
