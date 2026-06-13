"use client";

import useSWR from "swr";
import Link from "next/link";
import { jobsApi } from "@/lib/api/jobs";
import { queuesApi } from "@/lib/api/queues";
import StatusBadge from "@/components/StatusBadge";
import ProgressTimingInfo from "@/components/ProgressTimingInfo";
import { Loader2, ListOrdered } from "lucide-react";

export default function DashboardPage() {
  const { data: jobs } = useSWR("/jobs", () => jobsApi.list(), { refreshInterval: 5000 });
  const { data: queue } = useSWR("/queues", () => queuesApi.list(), { refreshInterval: 5000 });

  const running = jobs?.filter((j) => j.status === "running") ?? [];
  const queued = jobs?.filter((j) => j.status === "queued") ?? [];
  const completed = jobs?.filter((j) => j.status === "completed") ?? [];
  const failed = jobs?.filter((j) => j.status === "failed") ?? [];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-[var(--muted)] mt-1">Overview of your LoRA training jobs</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "Running", value: running.length, color: "text-blue-400" },
          { label: "Queued", value: queued.length, color: "text-yellow-400" },
          { label: "Completed", value: completed.length, color: "text-green-400" },
          { label: "Failed", value: failed.length, color: "text-red-400" },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-5">
            <div className={`text-3xl font-bold ${color}`}>{value}</div>
            <div className="text-[var(--muted)] text-sm mt-1">{label}</div>
          </div>
        ))}
      </div>

      {/* Active job */}
      {running.length > 0 && (
        <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Active Training</h2>
          {running.map((job) => {
            const pct =
              job.progress_step != null && job.progress_total != null && job.progress_total > 0
                ? Math.round((job.progress_step / job.progress_total) * 100)
                : null;
            return (
              <div key={job.id} className="space-y-3">
                <div className="flex items-center justify-between">
                  <Link href={`/jobs/${job.id}`} className="font-medium text-white hover:text-[var(--accent)]">
                    {job.name}
                  </Link>
                  <StatusBadge status={job.status} />
                </div>
                {pct != null && (
                  <div>
                    <div className="flex justify-between text-xs text-[var(--muted)] mb-1">
                      <span>
                        {job.progress_epoch != null && job.progress_epoch_total != null && (
                          <>epoch {job.progress_epoch}/{job.progress_epoch_total} · </>
                        )}
                        step {job.progress_step} / {job.progress_total}
                      </span>
                      <span>{pct}%</span>
                    </div>
                    <div className="bg-[var(--border)] rounded-full h-2">
                      <div
                        className="bg-[var(--accent)] h-2 rounded-full transition-all duration-500"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <ProgressTimingInfo
                      step={job.progress_step}
                      total={job.progress_total}
                      active
                      compact
                      className="mt-1"
                    />
                    {job.progress_avr_loss != null && (
                      <div className="text-xs text-[var(--muted)] mt-1">avr_loss {job.progress_avr_loss.toFixed(4)}</div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Queue */}
      {(queue?.length ?? 0) > 0 && (
        <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-6">
          <div className="flex items-center gap-2 mb-4">
            <ListOrdered size={18} className="text-[var(--muted)]" />
            <h2 className="text-lg font-semibold text-white">Queue</h2>
            <span className="ml-auto text-xs text-[var(--muted)]">{queue?.length} job(s)</span>
          </div>
          <div className="space-y-2">
            {queue?.map(({ entry, job }) => (
              <div key={entry.id} className="flex items-center gap-3 text-sm">
                <span className="w-6 h-6 rounded-full bg-[var(--border)] text-[var(--muted)] text-xs flex items-center justify-center shrink-0">
                  {entry.position}
                </span>
                <Link href={`/jobs/${job.id}`} className="text-white hover:text-[var(--accent)]">
                  {job.name}
                </Link>
                <StatusBadge status={job.status} />
              </div>
            ))}
          </div>
        </div>
      )}

      {running.length === 0 && (queue?.length ?? 0) === 0 && (
        <div className="text-center py-16 text-[var(--muted)]">
          No active jobs.{" "}
          <Link href="/jobs/new" className="text-[var(--accent)] hover:underline">
            Start training
          </Link>
        </div>
      )}
    </div>
  );
}
