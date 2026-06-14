"use client";

import useSWR from "swr";
import Link from "next/link";
import JobProgressBar from "@/components/JobProgressBar";
import LiveLogsPanel from "@/components/LiveLogsPanel";
import { jobsApi } from "@/lib/api/jobs";
import type { Job } from "@/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

interface SamplingJobPanelProps {
  job: Job;
}

function progressPercent(step: number | null, total: number | null): number | null {
  if (step == null || total == null || total <= 0) {
    return null;
  }
  return Math.round((step / total) * 100);
}

function JobSamples({ jobId, status }: { jobId: number; status: string }) {
  const { data } = useSWR(
    status === "completed" ? `/jobs/${jobId}/samples` : null,
    () => jobsApi.getSamples(jobId),
  );

  if (!data?.samples.length) {
    return null;
  }

  return (
    <div className="space-y-2">
      <h2 className="text-sm font-medium text-[var(--muted)]">Samples</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {data.samples.map((sample) => (
          <a key={sample.path} href={`${API_BASE_URL}${sample.url}`} target="_blank" rel="noreferrer">
            <img
              src={`${API_BASE_URL}${sample.url}`}
              alt={sample.filename}
              className="rounded-lg border border-[var(--border)] object-cover aspect-square"
            />
          </a>
        ))}
      </div>
    </div>
  );
}

export default function SamplingJobPanel({ job }: SamplingJobPanelProps) {
  const id = job.id;
  const sampling = job.sampling;
  const isRunning = job.status === "running";
  const showLogs =
    isRunning || job.status === "completed" || job.status === "failed" || job.status === "cancelled";
  const percent = progressPercent(job.progress_step, job.progress_total);

  return (
    <div className="space-y-6">
      {(isRunning || sampling?.progress_status) && (
        <JobProgressBar
          title={sampling?.progress_status ?? "Sampling"}
          step={job.progress_step}
          total={job.progress_total}
          percent={percent ?? 0}
          active={isRunning}
          barClassName="bg-purple-500"
          showSpinner={isRunning}
          showBar={percent != null}
          headerRight={
            percent != null ? (
              <span className="text-[var(--muted)]">
                step {job.progress_step} / {job.progress_total} ({percent}%)
              </span>
            ) : undefined
          }
        />
      )}

      {job.status === "failed" && job.error_message && (
        <div className="rounded-lg bg-red-900/30 border border-red-800 text-red-300 px-4 py-3 text-sm">
          <strong>Error:</strong> {job.error_message}
        </div>
      )}

      <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-4 space-y-3">
        <div className="text-xs text-[var(--muted)]">
          LoRA files ({sampling?.lora_paths.length ?? 0})
        </div>
        <ul className="space-y-1">
          {sampling?.lora_paths.map((path) => (
            <li key={path}>
              <code className="text-green-400 text-sm break-all">{path}</code>
            </li>
          ))}
        </ul>
        {sampling?.source_job_id != null && (
          <div>
            <div className="text-xs text-[var(--muted)] mb-1">Source Job</div>
            <Link href={`/jobs/${sampling.source_job_id}`} className="text-purple-400 text-sm hover:underline">
              Job #{sampling.source_job_id}
            </Link>
          </div>
        )}
        {job.output_path && (
          <div>
            <div className="text-xs text-[var(--muted)] mb-1">Output</div>
            <code className="text-green-400 text-sm break-all">{job.output_path}</code>
          </div>
        )}
      </div>

      {showLogs && (
        <LiveLogsPanel
          swrKey={`/jobs/${id}/logs`}
          fetcher={() => jobsApi.getLogs(id, 500)}
          isRunning={isRunning}
          showLogs={showLogs}
          status={job.status}
          title="Sampling Logs"
        />
      )}

      <JobSamples jobId={id} status={job.status} />
    </div>
  );
}
