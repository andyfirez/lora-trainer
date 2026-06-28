"use client";

import useSWR from "swr";
import Link from "next/link";
import JobProgressBar from "@/components/JobProgressBar";
import LiveLogsPanel from "@/components/LiveLogsPanel";
import { jobsApi } from "@/lib/api/jobs";
import type { Job } from "@/types";

interface TaggingJobPanelProps {
  job: Job;
}

function progressPercent(step: number | null, total: number | null): number | null {
  if (step == null || total == null || total <= 0) {
    return null;
  }
  return Math.round((step / total) * 100);
}

export default function TaggingJobPanel({ job }: TaggingJobPanelProps) {
  const id = job.id;
  const isRunning = job.status === "running";
  const showLogs =
    isRunning || job.status === "completed" || job.status === "failed" || job.status === "cancelled";
  const percent = progressPercent(job.progress_step, job.progress_total);
  const statusText = job.tagging?.progress_status ?? "Tagging";

  return (
    <div className="space-y-6">
      {(isRunning || job.status === "completed" || percent != null) && (
        <JobProgressBar
          title={statusText}
          step={job.progress_step}
          total={job.progress_total}
          percent={percent ?? (job.status === "completed" ? 100 : 0)}
          active={isRunning}
          barClassName="bg-[var(--accent)]"
          showSpinner={isRunning}
          showBar={percent != null || job.status === "completed"}
          headerRight={
            percent != null ? (
              <span className="text-[var(--muted)]">
                {job.progress_step} / {job.progress_total} ({percent}%)
              </span>
            ) : (
              <span className="text-[var(--muted)] capitalize">{job.status}</span>
            )
          }
        />
      )}

      {job.status === "failed" && job.error_message && (
        <div className="rounded-lg bg-red-900/30 border border-red-800 text-red-300 px-4 py-3 text-sm">
          <strong>Error:</strong> {job.error_message}
        </div>
      )}

      {job.tagging?.dataset_id != null && (
        <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-4">
          <div className="text-xs text-[var(--muted)] mb-1">Dataset</div>
          <Link
            href={`/datasets/${job.tagging.dataset_id}`}
            className="text-[var(--accent)] text-sm hover:underline"
          >
            Open dataset #{job.tagging.dataset_id}
          </Link>
        </div>
      )}

      {showLogs && id != null && (
        <LiveLogsPanel
          swrKey={`/jobs/${id}/logs`}
          fetcher={() => jobsApi.getLogs(id, 500)}
          isRunning={isRunning}
          showLogs={showLogs}
          status={job.status}
          title="Tagging Logs"
        />
      )}
    </div>
  );
}
