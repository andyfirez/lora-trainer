"use client";

import useSWR, { mutate as globalMutate } from "swr";
import Link from "next/link";
import { Loader2, Play, X, Trash2, ChevronUp } from "lucide-react";
import { jobsApi } from "@/lib/api/jobs";
import { queuesApi } from "@/lib/api/queues";
import StatusBadge from "@/components/StatusBadge";
import ProgressTimingInfo from "@/components/ProgressTimingInfo";
import type { Job } from "@/types";

const fetcher = () => jobsApi.list();

export default function JobsTable() {
  const { data: jobs, isLoading, mutate } = useSWR("/jobs", fetcher, { refreshInterval: 5000 });
  const { data: queue, mutate: mutateQueue } = useSWR("/queues", () => queuesApi.list(), { refreshInterval: 5000 });

  const refresh = () => { mutate(); mutateQueue(); };

  const handleEnqueue = async (job: Job) => {
    await jobsApi.enqueue(job.id);
    refresh();
  };

  const handleCancel = async (job: Job) => {
    await jobsApi.cancel(job.id);
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
      <div className="flex items-center justify-center py-20 text-[var(--muted)]">
        <Loader2 className="animate-spin mr-2" size={18} /> Loading jobs…
      </div>
    );
  }

  if (!jobs?.length) {
    return (
      <div className="text-center py-20 text-[var(--muted)]">
        No jobs yet.{" "}
        <Link href="/jobs/new" className="text-[var(--accent)] hover:underline">
          Create one
        </Link>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-[var(--border)]">
      <table className="w-full text-sm">
        <thead className="bg-[var(--surface)]">
          <tr>
            <th className="px-4 py-3 text-left text-[var(--muted)] font-medium">Name</th>
            <th className="px-4 py-3 text-left text-[var(--muted)] font-medium">Status</th>
            <th className="px-4 py-3 text-left text-[var(--muted)] font-medium">Progress</th>
            <th className="px-4 py-3 text-left text-[var(--muted)] font-medium">Created</th>
            <th className="px-4 py-3 text-right text-[var(--muted)] font-medium">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--border)]">
          {jobs.map((job) => {
            const progress =
              job.progress_step != null && job.progress_total != null && job.progress_total > 0
                ? Math.round((job.progress_step / job.progress_total) * 100)
                : null;
            const inQueue = queuedIds.has(job.id);

            return (
              <tr key={job.id} className="hover:bg-white/[0.02] transition-colors">
                <td className="px-4 py-3">
                  <Link href={`/jobs/${job.id}`} className="text-white hover:text-[var(--accent)] font-medium">
                    {job.name}
                  </Link>
                </td>
                <td className="px-4 py-3">
                  <StatusBadge status={job.status} />
                </td>
                <td className="px-4 py-3">
                  {job.status === "running" && progress != null ? (
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-[var(--border)] rounded-full h-1.5 w-24">
                          <div
                            className="bg-[var(--accent)] h-1.5 rounded-full transition-all"
                            style={{ width: `${progress}%` }}
                          />
                        </div>
                        <span className="text-[var(--muted)] text-xs">{progress}%</span>
                      </div>
                      {job.progress_avr_loss != null && (
                        <div className="text-[var(--muted)] text-xs">loss {job.progress_avr_loss.toFixed(4)}</div>
                      )}
                      <ProgressTimingInfo
                        step={job.progress_step}
                        total={job.progress_total}
                        active={job.status === "running"}
                        compact
                      />
                    </div>
                  ) : (
                    <span className="text-[var(--muted)]">—</span>
                  )}
                </td>
                <td className="px-4 py-3 text-[var(--muted)]">
                  {new Date(job.created_at).toLocaleDateString()}
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center justify-end gap-1">
                    {(job.status === "pending" || job.status === "failed" || job.status === "cancelled") && (
                      <button
                        onClick={() => handleEnqueue(job)}
                        title="Add to queue"
                        className="p-1.5 rounded hover:bg-white/10 text-green-400 hover:text-green-300"
                      >
                        <Play size={14} />
                      </button>
                    )}
                    {inQueue && job.status === "queued" && (
                      <button
                        onClick={() => handleMoveToTop(job.id)}
                        title="Move to top of queue"
                        className="p-1.5 rounded hover:bg-white/10 text-yellow-400 hover:text-yellow-300"
                      >
                        <ChevronUp size={14} />
                      </button>
                    )}
                    {(job.status === "queued" || job.status === "pending" || job.status === "running") && (
                      <button
                        onClick={() => handleCancel(job)}
                        title={job.status === "running" ? "Stop training" : "Cancel"}
                        className="p-1.5 rounded hover:bg-white/10 text-red-400 hover:text-red-300"
                      >
                        <X size={14} />
                      </button>
                    )}
                    {job.status !== "running" && (
                      <button
                        onClick={() => handleDelete(job)}
                        title="Delete"
                        className="p-1.5 rounded hover:bg-white/10 text-red-400 hover:text-red-300"
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
  );
}
