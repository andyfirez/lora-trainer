"use client";

import useSWR from "swr";
import Link from "next/link";
import JobProgressBar from "@/components/JobProgressBar";
import LiveLogsPanel from "@/components/LiveLogsPanel";
import SweepGridViewer from "@/components/sweep/SweepGridViewer";
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

function LegacySamples({ jobId, status }: { jobId: number; status: string }) {
  const { data } = useSWR(
    status === "completed" ? `/jobs/${jobId}/samples` : null,
    () => jobsApi.getSamples(jobId),
  );

  const legacy = data?.samples.filter((s) => s.kind === "legacy") ?? [];
  if (!legacy.length) return null;

  return (
    <div className="space-y-2">
      <h2 className="text-sm font-medium text-muted">Samples</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {legacy.map((sample) => (
          <a key={sample.path} href={`${API_BASE_URL}${sample.url}`} target="_blank" rel="noreferrer">
            <img
              src={`${API_BASE_URL}${sample.url}`}
              alt={sample.filename}
              className="rounded-lg border border-border object-cover aspect-square"
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
          barClassName="bg-sampling"
          showSpinner={isRunning}
          showBar={percent != null}
          headerRight={
            percent != null ? (
              <span className="text-muted">
                step {job.progress_step} / {job.progress_total} ({percent}%)
              </span>
            ) : undefined
          }
        />
      )}

      {sampling?.lora_paths?.length ? (
        <div className="space-y-1">
          <h2 className="text-sm font-medium text-muted">LoRA paths</h2>
          <ul className="text-sm text-text space-y-1 font-mono break-all">
            {sampling.lora_paths.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {sampling?.source_job_id != null && (
        <p className="text-sm text-muted">
          Source training job:{" "}
          <Link href={`/jobs/${sampling.source_job_id}`} className="text-accent hover:underline">
            #{sampling.source_job_id}
          </Link>
        </p>
      )}

      {job.status === "completed" && (
        <>
          <SweepGridViewer jobId={id} status={job.status} />
          <LegacySamples jobId={id} status={job.status} />
        </>
      )}

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
    </div>
  );
}
