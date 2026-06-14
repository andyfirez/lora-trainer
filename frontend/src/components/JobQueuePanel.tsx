"use client";

import useSWR from "swr";
import Link from "next/link";
import { Loader2, Play, X, ChevronUp } from "lucide-react";
import { jobsApi } from "@/lib/api/jobs";
import { queuesApi } from "@/lib/api/queues";
import StatusBadge from "@/components/StatusBadge";
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
      <div className="flex items-center justify-center py-8 text-[var(--muted)]">
        <Loader2 className="animate-spin mr-2" size={18} /> Loading queue…
      </div>
    );
  }

  if (!queue?.length) {
    return (
      <div className="text-center py-8 text-[var(--muted)] rounded-xl border border-[var(--border)]">
        Queue is empty. Run a job from a config to add it here.
      </div>
    );
  }

  return (
    <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] overflow-hidden">
      <div className="divide-y divide-[var(--border)]">
        {queue.map(({ entry, job }) => {
          const isSampling = job.job_type === "sampling";
          const linkClass = isSampling ? "hover:text-purple-400" : "hover:text-[var(--accent)]";

          return (
            <div key={entry.id} className="flex items-center gap-3 px-4 py-3 text-sm">
              <span className="w-6 h-6 rounded-full bg-[var(--border)] text-[var(--muted)] text-xs flex items-center justify-center shrink-0">
                {entry.position}
              </span>
              <Link href={`/jobs/${job.id}`} className={`text-white font-medium ${linkClass}`}>
                {job.name}
              </Link>
              <span className="text-xs text-[var(--muted)]">{job.job_type}</span>
              <StatusBadge status={job.status} />
              <div className="ml-auto flex items-center gap-1">
                {(job.status === "pending" || job.status === "failed" || job.status === "cancelled") && (
                  <button
                    onClick={() => void handleEnqueue(job)}
                    title="Add to queue"
                    className="p-1.5 rounded hover:bg-white/10 text-green-400 hover:text-green-300"
                  >
                    <Play size={14} />
                  </button>
                )}
                {job.status === "queued" && (
                  <button
                    onClick={() => void handleMoveToTop(job.id)}
                    title="Move to top of queue"
                    className="p-1.5 rounded hover:bg-white/10 text-yellow-400 hover:text-yellow-300"
                  >
                    <ChevronUp size={14} />
                  </button>
                )}
                {(job.status === "queued" || job.status === "pending" || job.status === "running") && (
                  <button
                    onClick={() => void handleCancel(job)}
                    title={job.status === "running" ? "Stop" : "Cancel"}
                    className="p-1.5 rounded hover:bg-white/10 text-red-400 hover:text-red-300"
                  >
                    <X size={14} />
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
