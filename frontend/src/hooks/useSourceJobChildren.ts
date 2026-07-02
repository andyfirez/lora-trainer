"use client";

import useSWR from "swr";
import { jobsApi } from "@/lib/api/jobs";
import type { Job } from "@/types";

export function useSourceJobChildren(sourceJobId: number | null) {
  const swrKey = sourceJobId != null ? `/jobs?source_job_id=${sourceJobId}` : null;
  const { data: childJobs, mutate, isLoading } = useSWR<Job[]>(
    swrKey,
    () => (sourceJobId != null ? jobsApi.list({ source_job_id: sourceJobId }) : Promise.resolve([])),
    {
      refreshInterval: (latest) =>
        latest?.some((job) => job.status === "running" || job.status === "queued") ? 1000 : 5000,
    },
  );

  const jobs = childJobs ?? [];
  const hasActiveSamplingJob = jobs.some(
    (job) =>
      job.job_type === "sampling" && (job.status === "queued" || job.status === "running"),
  );

  return {
    childJobs: jobs,
    hasActiveSamplingJob,
    isLoading,
    mutate,
  };
}
