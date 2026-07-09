"use client";

import useSWR from "swr";
import Link from "next/link";
import { Loader2, Play, X, Trash2, ChevronUp } from "lucide-react";
import { jobsApi } from "@/lib/api/jobs";
import { queuesApi } from "@/lib/api/queues";
import StatusBadge from "@/components/StatusBadge";
import ProgressTimingInfo from "@/components/ProgressTimingInfo";
import { Button, Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui";
import { jobLinkClass, jobBarClass, actionIconClass } from "@/lib/jobTypeColors";
import { cn } from "@/lib/cn";
import type { Job } from "@/types";

interface JobsListProps {
  jobType?: import("@/types").JobType;
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

  const handleEnqueue = async (job: Job) => {
    await jobsApi.enqueue(job.id);
    refresh();
  };

  const handleCancel = async (job: Job) => {
    if (job.status === "running" && job.job_type === "training") {
      const saveCheckpoint = window.confirm("Save checkpoint before stopping this job?");
      await jobsApi.cancel(job.id, saveCheckpoint);
    } else {
      await jobsApi.cancel(job.id);
    }
    refresh();
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
      <div className="flex items-center justify-center py-20 text-text-muted">
        <Loader2 className="animate-spin mr-2" size={18} /> Loading jobs…
      </div>
    );
  }

  if (!jobs?.length) {
    return (
      <div className="text-center py-20 text-text-muted rounded-xl border border-border">
        No jobs yet.{" "}
        <Link href="/configs" className="text-accent hover:underline">
          Create a config and run a job
        </Link>
      </div>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead>Type</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Progress</TableHead>
          <TableHead>Created</TableHead>
          <TableHead align="right">Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {jobs.map((job) => {
          const progress =
            job.progress_step != null && job.progress_total != null && job.progress_total > 0
              ? Math.round((job.progress_step / job.progress_total) * 100)
              : null;
          const inQueue = queuedIds.has(job.id);
          const isSampling = job.job_type === "sampling";

          return (
            <TableRow key={job.id}>
              <TableCell>
                <Link
                  href={`/jobs/${job.id}`}
                  className={cn("text-text font-medium", jobLinkClass(job.job_type))}
                >
                  {job.name}
                </Link>
              </TableCell>
              <TableCell className="text-text-muted capitalize">{job.job_type}</TableCell>
              <TableCell>
                <StatusBadge status={job.status} />
              </TableCell>
              <TableCell>
                {job.status === "running" && progress != null ? (
                  <div className="space-y-1">
                    {isSampling && job.sampling?.progress_status && (
                      <div className="text-xs text-text-muted truncate max-w-[200px]">
                        {job.sampling.progress_status}
                      </div>
                    )}
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-border rounded-full h-1.5 w-24">
                        <div
                          className={cn(jobBarClass(job.job_type), "h-1.5 rounded-full transition-all")}
                          style={{ width: `${progress}%` }}
                        />
                      </div>
                      <span className="text-text-muted text-xs">{progress}%</span>
                    </div>
                    {!isSampling && job.training?.progress_avr_loss != null && (
                      <div className="text-text-muted text-xs">
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
                  <span className="text-text-muted">—</span>
                )}
              </TableCell>
              <TableCell className="text-text-muted">
                {new Date(job.created_at).toLocaleDateString()}
              </TableCell>
              <TableCell>
                <div className="flex items-center justify-end gap-1">
                  {(job.status === "pending" || job.status === "failed" || job.status === "cancelled") && (
                    <Button
                      variant="icon"
                      onClick={() => void handleEnqueue(job)}
                      title="Add to queue"
                      className={actionIconClass("play")}
                    >
                      <Play size={14} />
                    </Button>
                  )}
                  {(job.status === "failed" || job.status === "cancelled") && job.can_resume && (
                    <Button
                      variant="icon"
                      onClick={() => void handleResume(job)}
                      title="Resume from latest checkpoint"
                      className={actionIconClass("resume")}
                    >
                      <Play size={14} />
                    </Button>
                  )}
                  {inQueue && job.status === "queued" && (
                    <Button
                      variant="icon"
                      onClick={() => void handleMoveToTop(job.id)}
                      title="Move to top of queue"
                      className={actionIconClass("queue")}
                    >
                      <ChevronUp size={14} />
                    </Button>
                  )}
                  {(job.status === "queued" || job.status === "pending" || job.status === "running") && (
                    <Button
                      variant="icon"
                      onClick={() => void handleCancel(job)}
                      title={job.status === "running" ? "Stop" : "Cancel"}
                      className={actionIconClass("danger")}
                    >
                      <X size={14} />
                    </Button>
                  )}
                  {job.status !== "running" && (
                    <Button
                      variant="danger"
                      onClick={() => void handleDelete(job)}
                      title="Delete"
                      className={actionIconClass("danger")}
                    >
                      <Trash2 size={14} />
                    </Button>
                  )}
                </div>
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}
