"use client";

import useSWR from "swr";
import { use } from "react";
import Link from "next/link";
import { ArrowLeft, Square, Loader2 } from "lucide-react";
import { samplingRunsApi } from "@/lib/api/samplingRuns";
import StatusBadge from "@/components/StatusBadge";
import JobProgressBar from "@/components/JobProgressBar";
import LiveLogsPanel from "@/components/LiveLogsPanel";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

interface Props {
  params: Promise<{ id: string }>;
}

function progressPercent(step: number | null, total: number | null): number | null {
  if (step == null || total == null || total <= 0) {
    return null;
  }
  return Math.round((step / total) * 100);
}

function SamplingSamples({ runId, status }: { runId: number; status: string }) {
  const { data } = useSWR(
    status === "completed" ? `/sampling-runs/${runId}/samples` : null,
    () => samplingRunsApi.getSamples(runId),
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

export default function SamplingRunDetailPage({ params }: Props) {
  const { id: idParam } = use(params);
  const id = Number(idParam);
  const { data: run, isLoading, mutate } = useSWR(
    `/sampling-runs/${id}`,
    () => samplingRunsApi.get(id),
    { refreshInterval: (latest) => (latest?.status === "running" ? 1000 : 0) },
  );

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-[var(--muted)] py-20">
        <Loader2 className="animate-spin" size={18} /> Loading…
      </div>
    );
  }

  if (!run) {
    return <div className="text-red-400">Sampling run not found</div>;
  }

  const isRunning = run.status === "running";
  const showLogs =
    isRunning || run.status === "completed" || run.status === "failed" || run.status === "cancelled";
  const percent = progressPercent(run.progress_step, run.progress_total);

  const handleCancel = async () => {
    await samplingRunsApi.cancel(id);
    mutate();
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center gap-3">
        <Link href="/jobs" className="p-2 rounded-lg hover:bg-white/5 text-[var(--muted)] hover:text-white">
          <ArrowLeft size={18} />
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-white">{run.name}</h1>
          <div className="flex items-center gap-3 mt-1">
            <StatusBadge status={run.status} />
            {run.pid && <span className="text-xs text-[var(--muted)]">PID {run.pid}</span>}
            {run.source_job_id != null && (
              <Link href={`/jobs/${run.source_job_id}`} className="text-xs text-purple-400 hover:underline">
                Source job #{run.source_job_id}
              </Link>
            )}
          </div>
        </div>
        {(run.status === "queued" || run.status === "pending" || run.status === "running") && (
          <button
            onClick={() => void handleCancel()}
            className="flex items-center gap-1.5 bg-red-700 hover:bg-red-600 text-white rounded-lg px-3 py-1.5 text-sm"
          >
            <Square size={13} /> {run.status === "running" ? "Stop Sampling" : "Cancel"}
          </button>
        )}
      </div>

      {(isRunning || run.progress_status) && (
        <JobProgressBar
          title={run.progress_status ?? "Sampling"}
          step={run.progress_step}
          total={run.progress_total}
          percent={percent ?? 0}
          active={isRunning}
          barClassName="bg-purple-500"
          showSpinner={isRunning}
          showBar={percent != null}
          headerRight={
            percent != null ? (
              <span className="text-[var(--muted)]">
                step {run.progress_step} / {run.progress_total} ({percent}%)
              </span>
            ) : undefined
          }
        />
      )}

      {run.status === "failed" && run.error_message && (
        <div className="rounded-lg bg-red-900/30 border border-red-800 text-red-300 px-4 py-3 text-sm">
          <strong>Error:</strong> {run.error_message}
        </div>
      )}

      <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-4 space-y-3">
        <div className="text-xs text-[var(--muted)]">LoRA files ({run.lora_paths.length})</div>
        <ul className="space-y-1">
          {run.lora_paths.map((path) => (
            <li key={path}>
              <code className="text-green-400 text-sm break-all">{path}</code>
            </li>
          ))}
        </ul>
        {run.output_path && (
          <div>
            <div className="text-xs text-[var(--muted)] mb-1">Output</div>
            <code className="text-green-400 text-sm break-all">{run.output_path}</code>
          </div>
        )}
      </div>

      {showLogs && (
        <LiveLogsPanel
          swrKey={`/sampling-runs/${id}/logs`}
          fetcher={() => samplingRunsApi.getLogs(id, 500)}
          isRunning={isRunning}
          showLogs={showLogs}
          status={run.status}
          title="Sampling Logs"
        />
      )}

      <SamplingSamples runId={id} status={run.status} />
    </div>
  );
}
