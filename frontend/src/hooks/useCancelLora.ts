"use client";

import { lorasApi } from "@/lib/api/loras";
import { useCancelRunnable } from "@/hooks/useCancelRunnable";
import type { LoraResponse } from "@/types";

export function useCancelLora(onSuccess: () => void) {
  const cancel = useCancelRunnable<LoraResponse>(lorasApi, onSuccess, {
    errorMessage: "Failed to stop training",
  });

  return {
    dialogLora: cancel.dialogRunnable,
    loading: cancel.loading,
    error: cancel.error,
    canSaveCheckpoint: cancel.canSaveCheckpoint,
    requestCancel: cancel.requestCancel,
    executeCancel: cancel.executeCancel,
    closeDialog: cancel.closeDialog,
  };
}
