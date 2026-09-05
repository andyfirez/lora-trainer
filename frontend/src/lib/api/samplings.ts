import { api, BASE_URL } from "@/lib/api/client";
import { createRunnableApi } from "@/lib/api/runnableApi";
import type {
  CreateSamplingRequest,
  ExperimentManifestResponse,
  PromptTokenCountResponse,
  SamplingOptionsResponse,
  SamplingProfile,
  SamplingResponse,
} from "@/types";

const runnableApi = createRunnableApi<SamplingResponse>("/samplings");

export const samplingsApi = {
  ...runnableApi,
  listExperiments: () => api.get<SamplingResponse[]>("/samplings/?kind=experiment"),
  create: (body: CreateSamplingRequest) => api.post<SamplingResponse>("/samplings/", body),
  getExperimentManifest: (id: number) =>
    api.get<ExperimentManifestResponse | null>(`/samplings/${id}/experiment-manifest`),
  getOptions: (profile: SamplingProfile = "comfy") =>
    api.get<SamplingOptionsResponse>(`/samplings/options?profile=${profile}`),
  tokenCount: (prompt: string, baseModelName: string) =>
    api.post<PromptTokenCountResponse>("/samplings/token-count", {
      prompt,
      base_model_name: baseModelName,
    }),
  livePreviewUrl: (id: number, cacheBust: number) =>
    `${BASE_URL}/samplings/${id}/live-preview?t=${cacheBust}`,
  generate: async (
    config: Record<string, unknown>,
    loraPaths?: string[],
    options: { kind?: CreateSamplingRequest["kind"]; name?: string; composeGrids?: boolean } = {},
  ) => {
    const withFlags = {
      ...config,
      compose_grids: options.composeGrids ?? config.compose_grids ?? false,
    };
    const created = await api.post<SamplingResponse>("/samplings/", {
      name: options.name ?? "Playground",
      config: withFlags,
      lora_paths: loraPaths,
      kind: options.kind ?? "playground",
    } satisfies CreateSamplingRequest);
    return runnableApi.enqueue(created.id);
  },
};
