"use client";

import { useCallback } from "react";
import type { RunnableResponse } from "@/types";

interface RunnableActionsApi<T extends RunnableResponse> {
  enqueue: (id: number) => Promise<T>;
  cancel: (id: number) => Promise<T>;
}

export function useRunnableActions<T extends RunnableResponse>(
  id: number,
  api: RunnableActionsApi<T>,
  mutate: () => void,
) {
  const handleEnqueue = useCallback(async () => {
    await api.enqueue(id);
    mutate();
  }, [api, id, mutate]);

  const handleCancel = useCallback(async () => {
    await api.cancel(id);
    mutate();
  }, [api, id, mutate]);

  return { handleEnqueue, handleCancel };
}
