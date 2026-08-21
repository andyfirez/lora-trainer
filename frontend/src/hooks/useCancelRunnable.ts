"use client";

import { useCallback, useState } from "react";
import { canSaveCheckpointOnStop, needsStopDialog } from "@/lib/runnable/cancel";
import type { RunnableResponse } from "@/types";

interface CancelApi {
  cancel: (id: number, saveCheckpoint?: boolean) => Promise<unknown>;
}

export function useCancelRunnable<T extends RunnableResponse>(
  api: CancelApi,
  onSuccess: () => void,
  options?: { errorMessage?: string },
) {
  const [dialogRunnable, setDialogRunnable] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const executeCancel = useCallback(
    async (runnable: T, saveCheckpoint: boolean) => {
      setLoading(true);
      setError(null);
      try {
        await api.cancel(runnable.id, saveCheckpoint);
        setDialogRunnable(null);
        onSuccess();
      } catch (err) {
        setError(err instanceof Error ? err.message : (options?.errorMessage ?? "Failed to stop job"));
      } finally {
        setLoading(false);
      }
    },
    [api, onSuccess, options?.errorMessage],
  );

  const requestCancel = useCallback(
    (runnable: T) => {
      if (needsStopDialog(runnable)) {
        setError(null);
        setDialogRunnable(runnable);
        return;
      }
      void executeCancel(runnable, false);
    },
    [executeCancel],
  );

  const closeDialog = useCallback(() => {
    if (!loading) {
      setDialogRunnable(null);
      setError(null);
    }
  }, [loading]);

  return {
    dialogRunnable,
    loading,
    error,
    canSaveCheckpoint: dialogRunnable ? canSaveCheckpointOnStop(dialogRunnable) : false,
    requestCancel,
    executeCancel,
    closeDialog,
  };
}
