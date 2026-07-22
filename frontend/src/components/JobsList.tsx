"use client";

import useSWR from "swr";
import Link from "next/link";
import { Loader2, Play, X, Trash2, ChevronUp } from "lucide-react";
import { jobsApi } from "@/lib/api/jobs";
import { queuesApi } from "@/lib/api/queues";
import StatusBadge from "@/components/StatusBadge";
import ProgressTimingInfo from "@/components/ProgressTimingInfo";
import StopJobDialog from "@/components/StopJobDialog";
import { useCancelJob } from "@/hooks/useCancelJob";
import type { Job, JobType } from "@/types";

interface JobsListProps {
  jobType?: JobType;
  sourceJobId?: number;
}

export default function JobsList({ jobType, sourceJobId }: JobsListProps) {
  const swrKey = sourceJobId != null
    ? `/jobs?source_job_id=${sourceJobId}`
    : jobType != null
      ? `/jobs?job_type=${jobType}`
      : "/jobs";

  const { data: jobs, isLoading, mutate } = useSWR(swrKey, () => jobsApi.list({ job_type: jobType, source_job_id: sourceJobId }), {
    refreshInterval: (latest) =>
      latest?.some((job) => job.status === "running") ? 1000 : 5000,
  });
  const { data: queue, mutate: mutateQueue } = useSWR("/queues", () => queuesApi.list(), { refreshInterval: 5000 });

  const refresh = () => {
    mutate();
    mutateQueue();
  };

  const {
    dialogJob,
    loading: cancelLoading,
    error: cancelError,
    canSaveCheckpoint,
    requestCancel,
    executeCancel,
    closeDialog,
  } = useCancelJob(refresh);

  const handleEnqueue = async (job: Job) => {
    await jobsApi.enqueue(job.id);
    refresh();
  };

  const handleCancel = (job: Job) => {
    requestCancel(job);
  };

  const handleResume = async (job: Job) => {
    await jobsApi.resume(job.id);
    refresh();
  };

  const handleDelete = async (job: Job) => {
    if (!confirm(`Delete job "${job.name}"?`)) return;
    await jobsApi.delete(job.id);
    refresh();
  };

  const handleMoveToTop = async (jobId: number) => {
    await queuesApi.moveToTop(jobId);
    refresh();
  };

  const queuedIds = new Set((queue ?? []).map((q) => q.entry.job_id));

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20 text-muted">
        <Loader2 className="animate-spin mr-2" size={18} /> Loading jobs…
      </div>
    );
  }

  if (!jobs?.length) {
    return (
      <div className="text-center py-20 text-muted rounded-xl border border-border">
        No jobs yet.{" "}
        <Link href="/trainings" className="text-accent hover:underline">
          Create a training config and run a job
        </Link>
      </div>
    );
  }

  return (
    <>
      <StopJobDialog
        open={dialogJob != null}
        jobName={dialogJob?.name ?? ""}
        canSaveCheckpoint={canSaveCheckpoint}
        loading={cancelLoading}
        error={cancelError}
        onClose={closeDialog}
        onStopNow={() => dialogJob && void executeCancel(dialogJob, false)}
        onSaveAndStop={() => dialogJob && void executeCancel(dialogJob, true)}
      />
    <div className="overflow-x-auto rounded-xl border border-border">
      <table className="w-full text-sm">
        <thead className="bg-surface">
          <tr>
            <th className="px-4 py-3 text-left text-muted font-medium">Name</th>
            <th className="px-4 py-3 text-left text-muted font-medium">Type</th>
            <th className="px-4 py-3 text-left text-muted font-medium">Status</th>
            <th className="px-4 py-3 text-left text-muted font-medium">Progress</th>
            <th className="px-4 py-3 text-left text-muted font-medium">Created</th>
            <th className="px-4 py-3 text-right text-muted font-medium">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {jobs.map((job) => {
            const progress =
              job.progress_step != null && job.progress_total != null && job.progress_total > 0
                ? Math.round((job.progress_step / job.progress_total) * 100)
                : null;
            const inQueue = queuedIds.has(job.id);
            const isSampling = job.job_type === "sampling";
            const linkClass = isSampling ? "hover:text-sampling" : "hover:text-accent";
            const barClass = isSampling ? "bg-sampling" : "bg-accent";

            return (
              <tr key={job.id} className="hover:bg-white/[0.02] transition-colors">
                <td className="px-4 py-3">
                  <Link href={`/jobs/${job.id}`} className={`text-text font-medium ${linkClass}`}>
                    {job.name}
                  </Link>
                </td>
                <td className="px-4 py-3 text-muted capitalize">{job.job_type}</td>
                <td className="px-4 py-3">
                  <StatusBadge status={job.status} />
                </td>
                <td className="px-4 py-3">
                  {job.status === "running" && progress != null ? (
                    <div className="space-y-1">
                      {isSampling && job.sampling?.progress_status && (
                        <div className="text-xs text-muted truncate max-w-[200px]">
                          {job.sampling.progress_status}
                        </div>
                      )}
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-border rounded-full h-1.5 w-24">
                          <div
                            className={`${barClass} h-1.5 rounded-full transition-all`}
                            style={{ width: `${progress}%` }}
                          />
                        </div>
                        <span className="text-muted text-xs">{progress}%</span>
                      </div>
                      {!isSampling && job.training?.progress_avr_loss != null && (
                        <div className="text-muted text-xs">
                          loss {job.training.progress_avr_loss.toFixed(4)}
                        </div>
                      )}
                      <ProgressTimingInfo
                        step={job.progress_step}
                        total={job.progress_total}
                        active={job.status === "running"}
                        compact
                      />
                    </div>
                  ) : (
                    <span className="text-muted">—</span>
                  )}
                </td>
                <td className="px-4 py-3 text-muted">
                  {new Date(job.created_at).toLocaleDateString()}
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center justify-end gap-1">
                    {(job.status === "pending" || job.status === "failed" || job.status === "cancelled") && (
                      <button
                        onClick={() => void handleEnqueue(job)}
                        title="Add to queue"
                        className="p-1.5 rounded hover:bg-white/10 text-success hover:text-success"
                      >
                        <Play size={14} />
                      </button>
                    )}
                    {(job.status === "failed" || job.status === "cancelled") && job.can_resume && (
                      <button
                        onClick={() => void handleResume(job)}
                        title="Resume from latest checkpoint"
                        className="p-1.5 rounded hover:bg-white/10 text-running hover:text-running"
                      >
                        <Play size={14} />
                      </button>
                    )}
                    {inQueue && job.status === "queued" && (
                      <button
                        onClick={() => void handleMoveToTop(job.id)}
                        title="Move to top of queue"
                        className="p-1.5 rounded hover:bg-white/10 text-warning hover:text-warning/80"
                      >
                        <ChevronUp size={14} />
                      </button>
                    )}
                    {(job.status === "queued" || job.status === "pending" || job.status === "running") && (
                      <button
                        onClick={() => handleCancel(job)}
                        disabled={cancelLoading}
                        title={job.status === "running" ? "Stop" : "Cancel"}
                        className="p-1.5 rounded hover:bg-white/10 text-error hover:text-error disabled:opacity-50"
                      >
                        <X size={14} />
                      </button>
                    )}
                    {job.status !== "running" && (
                      <button
                        onClick={() => void handleDelete(job)}
                        title="Delete"
                        className="p-1.5 rounded hover:bg-white/10 text-error hover:text-error"
                      >
                        <Trash2 size={14} />
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
    </>
  );
}
