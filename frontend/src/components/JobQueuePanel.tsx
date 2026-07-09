"use client";

import useSWR from "swr";
import Link from "next/link";
import { Loader2, Play, X, ChevronUp } from "lucide-react";
import { jobsApi } from "@/lib/api/jobs";
import { queuesApi } from "@/lib/api/queues";
import StatusBadge from "@/components/StatusBadge";
import { Button } from "@/components/ui";
import { jobLinkClass, actionIconClass } from "@/lib/jobTypeColors";
import { cn } from "@/lib/cn";
import type { Job } from "@/types";

export default function JobQueuePanel() {
  const { data: queue, isLoading, mutate } = useSWR("/queues", () => queuesApi.list(), { refreshInterval: 5000 });

  const refresh = () => mutate();

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

  const handleMoveToTop = async (jobId: number) => {
    await queuesApi.moveToTop(jobId);
    refresh();
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8 text-text-muted">
        <Loader2 className="animate-spin mr-2" size={18} /> Loading queue…
      </div>
    );
  }

  if (!queue?.length) {
    return (
      <div className="text-center py-8 text-text-muted rounded-xl border border-border">
        Queue is empty. Run a job from a config to add it here.
      </div>
    );
  }

  return (
    <div className="bg-surface rounded-xl border border-border overflow-hidden">
      <div className="divide-y divide-border">
        {queue.map(({ entry, job }) => (
          <div key={entry.id} className="flex items-center gap-3 px-4 py-3 text-sm">
            <span className="w-6 h-6 rounded-full bg-border text-text-muted text-xs flex items-center justify-center shrink-0">
              {entry.position}
            </span>
            <Link
              href={`/jobs/${job.id}`}
              className={cn("text-text font-medium", jobLinkClass(job.job_type))}
            >
              {job.name}
            </Link>
            <span className="text-xs text-text-muted">{job.job_type}</span>
            <StatusBadge status={job.status} />
            <div className="ml-auto flex items-center gap-1">
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
              {job.status === "queued" && (
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
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
