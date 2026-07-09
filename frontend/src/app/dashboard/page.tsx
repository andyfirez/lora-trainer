"use client";

import useSWR from "swr";
import Link from "next/link";
import { jobsApi } from "@/lib/api/jobs";
import { queuesApi } from "@/lib/api/queues";
import StatusBadge from "@/components/StatusBadge";
import ProgressTimingInfo from "@/components/ProgressTimingInfo";
import { ListOrdered } from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import Card, { CardTitle } from "@/components/ui/Card";
import { cn } from "@/lib/cn";

const STAT_CARDS = [
  { label: "Running", key: "running" as const, color: "text-running", delay: "" },
  { label: "Queued", key: "queued" as const, color: "text-warning", delay: "animate-fade-up-delay-1" },
  { label: "Completed", key: "completed" as const, color: "text-success", delay: "animate-fade-up-delay-2" },
  { label: "Failed", key: "failed" as const, color: "text-error", delay: "animate-fade-up-delay-3" },
];

export default function DashboardPage() {
  const { data: jobs } = useSWR("/jobs", () => jobsApi.list(), {
    refreshInterval: (latest) => (latest?.some((j) => j.status === "running") ? 1000 : 5000),
  });
  const { data: queue } = useSWR("/queues", () => queuesApi.list(), { refreshInterval: 5000 });

  const counts = {
    running: jobs?.filter((j) => j.status === "running").length ?? 0,
    queued: jobs?.filter((j) => j.status === "queued").length ?? 0,
    completed: jobs?.filter((j) => j.status === "completed").length ?? 0,
    failed: jobs?.filter((j) => j.status === "failed").length ?? 0,
  };
  const runningJobs = jobs?.filter((j) => j.status === "running") ?? [];

  return (
    <div className="space-y-8">
      <PageHeader
        title="Dashboard"
        description="Overview of training and sampling jobs"
      />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {STAT_CARDS.map(({ label, key, color, delay }) => (
          <Card key={label} className={cn("animate-fade-up", delay)}>
            <div className={cn("text-3xl font-bold font-display", color)}>{counts[key]}</div>
            <div className="text-muted text-sm mt-1">{label}</div>
          </Card>
        ))}
      </div>

      {runningJobs.length > 0 && (
        <Card padding="lg" className="space-y-6">
          <CardTitle>Active Jobs</CardTitle>
          {runningJobs.map((job) => {
            const pct =
              job.progress_step != null && job.progress_total != null && job.progress_total > 0
                ? Math.round((job.progress_step / job.progress_total) * 100)
                : null;
            const isSampling = job.job_type === "sampling";
            const barClass = isSampling ? "bg-sampling" : "bg-accent";
            const linkClass = isSampling ? "hover:text-sampling" : "hover:text-accent";

            return (
              <div key={job.id} className="space-y-3">
                <div className="flex items-center justify-between">
                  <Link href={`/jobs/${job.id}`} className={cn("font-medium text-text", linkClass)}>
                    {job.name}
                  </Link>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted capitalize">{job.job_type}</span>
                    <StatusBadge status={job.status} />
                  </div>
                </div>
                {isSampling && job.sampling?.progress_status && (
                  <div className="text-xs text-muted">{job.sampling.progress_status}</div>
                )}
                {pct != null && (
                  <div>
                    <div className="flex justify-between text-xs text-muted mb-1">
                      <span>
                        {!isSampling &&
                          job.training?.progress_epoch != null &&
                          job.training.progress_epoch_total != null && (
                            <>epoch {job.training.progress_epoch}/{job.training.progress_epoch_total} · </>
                          )}
                        step {job.progress_step} / {job.progress_total}
                      </span>
                      <span>{pct}%</span>
                    </div>
                    <div className="bg-border rounded-full h-2">
                      <div
                        className={cn(barClass, "h-2 rounded-full transition-all duration-500")}
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
                    {!isSampling && job.training?.progress_avr_loss != null && (
                      <div className="text-xs text-muted mt-1">
                        avr_loss {job.training.progress_avr_loss.toFixed(4)}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </Card>
      )}

      {(queue?.length ?? 0) > 0 && (
        <Card padding="lg">
          <div className="flex items-center gap-2 mb-4">
            <ListOrdered size={18} className="text-muted" />
            <CardTitle className="mb-0">Queue</CardTitle>
            <span className="ml-auto text-xs text-muted">{queue?.length} item(s)</span>
          </div>
          <div className="space-y-2">
            {queue?.map(({ entry, job }) => {
              const isSampling = job.job_type === "sampling";
              const linkClass = isSampling ? "hover:text-sampling" : "hover:text-accent";

              return (
                <div key={entry.id} className="flex items-center gap-3 text-sm">
                  <span className="w-6 h-6 rounded-full bg-border text-muted text-xs flex items-center justify-center shrink-0">
                    {entry.position}
                  </span>
                  <Link href={`/jobs/${job.id}`} className={cn("text-text", linkClass)}>
                    {job.name}
                  </Link>
                  <span className="text-xs text-muted capitalize">{job.job_type}</span>
                  <StatusBadge status={job.status} />
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {runningJobs.length === 0 && (queue?.length ?? 0) === 0 && (
        <div className="text-center py-16 text-muted">
          No active jobs.{" "}
          <Link href="/configs" className="text-accent hover:underline">
            Create a config and run a job
          </Link>
        </div>
      )}
    </div>
  );
}
