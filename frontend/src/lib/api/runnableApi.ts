import { api } from "@/lib/api/client";
import type { LogsResponse, RunnableResponse, RunnableSamplesResponse } from "@/types";

interface CreateRunnableApiOptions {
  cancelWithCheckpoint?: boolean;
}

export function createRunnableApi<T extends RunnableResponse>(
  basePath: string,
  options: CreateRunnableApiOptions = {},
) {
  const cancel = options.cancelWithCheckpoint
    ? (id: number, saveCheckpoint = false) =>
        api.post<T>(`${basePath}/${id}/cancel`, { save_checkpoint: saveCheckpoint })
    : (id: number) => api.post<T>(`${basePath}/${id}/cancel`);

  return {
    list: () => api.get<T[]>(`${basePath}/`),
    get: (id: number) => api.get<T>(`${basePath}/${id}`),
    enqueue: (id: number) => api.post<T>(`${basePath}/${id}/enqueue`),
    cancel,
    getLogs: (id: number, tail = 500) => api.get<LogsResponse>(`${basePath}/${id}/logs?tail=${tail}`),
    getSamples: (id: number) => api.get<RunnableSamplesResponse>(`${basePath}/${id}/samples`),
  };
}
