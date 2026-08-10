"use client";

import { useCallback, useState } from "react";
import { lorasApi } from "@/lib/api/loras";
import { canSaveCheckpointOnStop, needsStopDialog } from "@/lib/runnable/cancel";
import type { LoraResponse } from "@/types";

export function useCancelLora(onSuccess: () => void) {
  const [dialogLora, setDialogLora] = useState<LoraResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const executeCancel = useCallback(
    async (lora: LoraResponse, saveCheckpoint: boolean) => {
      setLoading(true);
      setError(null);
      try {
        await lorasApi.cancel(lora.id, saveCheckpoint);
        setDialogLora(null);
        onSuccess();
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to stop training");
      } finally {
        setLoading(false);
      }
    },
    [onSuccess],
  );

  const requestCancel = useCallback(
    (lora: LoraResponse) => {
      if (needsStopDialog(lora)) {
        setError(null);
        setDialogLora(lora);
        return;
      }
      void executeCancel(lora, false);
    },
    [executeCancel],
  );

  const closeDialog = useCallback(() => {
    if (!loading) {
      setDialogLora(null);
      setError(null);
    }
  }, [loading]);

  return {
    dialogLora,
    loading,
    error,
    canSaveCheckpoint: dialogLora ? canSaveCheckpointOnStop(dialogLora) : false,
    requestCancel,
    executeCancel,
    closeDialog,
  };
}
