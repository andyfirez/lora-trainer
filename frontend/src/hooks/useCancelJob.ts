"use client";

import { useCallback, useState } from "react";
import { jobsApi } from "@/lib/api/jobs";
import { canSaveCheckpointOnStop, needsStopDialog } from "@/lib/jobs/cancel";
import type { Job } from "@/types";

export function useCancelJob(onSuccess: () => void) {
  const [dialogJob, setDialogJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const executeCancel = useCallback(
    async (job: Job, saveCheckpoint: boolean) => {
      setLoading(true);
      setError(null);
      try {
        await jobsApi.cancel(job.id, saveCheckpoint);
        setDialogJob(null);
        onSuccess();
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to stop job");
      } finally {
        setLoading(false);
      }
    },
    [onSuccess],
  );

  const requestCancel = useCallback(
    (job: Job) => {
      if (needsStopDialog(job)) {
        setError(null);
        setDialogJob(job);
        return;
      }
      void executeCancel(job, false);
    },
    [executeCancel],
  );

  const closeDialog = useCallback(() => {
    if (!loading) {
      setDialogJob(null);
      setError(null);
    }
  }, [loading]);

  return {
    dialogJob,
    loading,
    error,
    canSaveCheckpoint: dialogJob ? canSaveCheckpointOnStop(dialogJob) : false,
    requestCancel,
    executeCancel,
    closeDialog,
  };
}
