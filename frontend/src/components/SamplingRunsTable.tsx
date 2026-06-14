"use client";

import useSWR from "swr";
import Link from "next/link";
import { Loader2, Play, X, ChevronUp } from "lucide-react";
import { samplingRunsApi } from "@/lib/api/samplingRuns";
import { queuesApi } from "@/lib/api/queues";
import StatusBadge from "@/components/StatusBadge";
import ProgressTimingInfo from "@/components/ProgressTimingInfo";
import type { SamplingRun } from "@/types";

const fetcher = () => samplingRunsApi.list();

export default function SamplingRunsTable() {
  const { data: runs, isLoading, mutate } = useSWR("/sampling-runs", fetcher, {
    refreshInterval: (latest) => (latest?.some((run) => run.status === "running") ? 1000 : 5000),
  });
  const { data: queue, mutate: mutateQueue } = useSWR("/queues", () => queuesApi.list(), { refreshInterval: 5000 });

  const refresh = () => {
    mutate();
    mutateQueue();
  };

  const handleCancel = async (run: SamplingRun) => {
    await samplingRunsApi.cancel(run.id);
    refresh();
  };

  const handleEnqueue = async (run: SamplingRun) => {
    await samplingRunsApi.enqueue(run.id);
    refresh();
  };

  const handleMoveToTop = async (runId: number) => {
    await queuesApi.moveToTop("sampling", runId);
    refresh();
  };

  const queuedIds = new Set(
    (queue ?? [])
      .filter((q) => q.entry.item_type === "sampling")
      .map((q) => q.entry.item_id),
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12 text-[var(--muted)]">
        <Loader2 className="animate-spin mr-2" size={18} /> Loading sampling runs…
      </div>
    );
  }

  if (!runs?.length) {
    return (
      <div className="text-center py-12 text-[var(--muted)] rounded-xl border border-[var(--border)]">
        No sampling runs yet. Start one from a training job&apos;s Sampling panel.
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
            <th className="px-4 py-3 text-left text-[var(--muted)] font-medium">Source Job</th>
            <th className="px-4 py-3 text-left text-[var(--muted)] font-medium">Created</th>
            <th className="px-4 py-3 text-right text-[var(--muted)] font-medium">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--border)]">
          {runs.map((run) => {
            const progress =
              run.progress_step != null && run.progress_total != null && run.progress_total > 0
                ? Math.round((run.progress_step / run.progress_total) * 100)
                : null;
            const inQueue = queuedIds.has(run.id);

            return (
              <tr key={run.id} className="hover:bg-white/[0.02] transition-colors">
                <td className="px-4 py-3">
                  <Link
                    href={`/sampling-runs/${run.id}`}
                    className="text-white hover:text-purple-400 font-medium"
                  >
                    {run.name}
                  </Link>
                </td>
                <td className="px-4 py-3">
                  <StatusBadge status={run.status} />
                </td>
                <td className="px-4 py-3">
                  {run.status === "running" && progress != null ? (
                    <div className="space-y-1">
                      {run.progress_status && (
                        <div className="text-xs text-[var(--muted)] truncate max-w-[200px]">{run.progress_status}</div>
                      )}
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-[var(--border)] rounded-full h-1.5 w-24">
                          <div
                            className="bg-purple-500 h-1.5 rounded-full transition-all"
                            style={{ width: `${progress}%` }}
                          />
                        </div>
                        <span className="text-[var(--muted)] text-xs">{progress}%</span>
                      </div>
                      <ProgressTimingInfo
                        step={run.progress_step}
                        total={run.progress_total}
                        active={run.status === "running"}
                        compact
                      />
                    </div>
                  ) : (
                    <span className="text-[var(--muted)]">—</span>
                  )}
                </td>
                <td className="px-4 py-3 text-[var(--muted)]">
                  {run.source_job_id != null ? (
                    <Link href={`/jobs/${run.source_job_id}`} className="hover:text-purple-400">
                      Job #{run.source_job_id}
                    </Link>
                  ) : (
                    "—"
                  )}
                </td>
                <td className="px-4 py-3 text-[var(--muted)]">
                  {new Date(run.created_at).toLocaleDateString()}
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center justify-end gap-1">
                    {(run.status === "pending" || run.status === "failed" || run.status === "cancelled") && (
                      <button
                        onClick={() => handleEnqueue(run)}
                        title="Add to queue"
                        className="p-1.5 rounded hover:bg-white/10 text-green-400 hover:text-green-300"
                      >
                        <Play size={14} />
                      </button>
                    )}
                    {inQueue && run.status === "queued" && (
                      <button
                        onClick={() => handleMoveToTop(run.id)}
                        title="Move to top of queue"
                        className="p-1.5 rounded hover:bg-white/10 text-yellow-400 hover:text-yellow-300"
                      >
                        <ChevronUp size={14} />
                      </button>
                    )}
                    {(run.status === "queued" || run.status === "pending" || run.status === "running") && (
                      <button
                        onClick={() => handleCancel(run)}
                        title={run.status === "running" ? "Stop sampling" : "Cancel"}
                        className="p-1.5 rounded hover:bg-white/10 text-red-400 hover:text-red-300"
                      >
                        <X size={14} />
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
