"use client";

import useSWR from "swr";
import JobProgressBar from "@/components/JobProgressBar";
import LiveLogsPanel from "@/components/LiveLogsPanel";
import SweepGridViewer from "@/components/sweep/SweepGridViewer";
import { samplingsApi } from "@/lib/api/samplings";
import type { SamplingResponse } from "@/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

interface SamplingRunPanelProps {
  sampling: SamplingResponse;
}

function progressPercent(step: number | null, total: number | null): number | null {
  if (step == null || total == null || total <= 0) {
    return null;
  }
  return Math.round((step / total) * 100);
}

function LegacySamples({ samplingId, status }: { samplingId: number; status: string }) {
  const { data } = useSWR(
    status === "completed" ? `/samplings/${samplingId}/samples` : null,
    () => samplingsApi.getSamples(samplingId),
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

export default function SamplingRunPanel({ sampling }: SamplingRunPanelProps) {
  const id = sampling.id;
  const isRunning = sampling.status === "running";
  const showLogs =
    isRunning || sampling.status === "completed" || sampling.status === "failed" || sampling.status === "cancelled";
  const percent = progressPercent(sampling.progress_step, sampling.progress_total);

  return (
    <div className="space-y-6">
      {(isRunning || sampling.progress_status) && (
        <JobProgressBar
          title={sampling.progress_status ?? "Sampling"}
          step={sampling.progress_step}
          total={sampling.progress_total}
          percent={percent ?? 0}
          active={isRunning}
          elapsedSeconds={sampling.elapsed_seconds}
          barClassName="bg-sampling"
          showSpinner={isRunning}
          showBar={percent != null}
          headerRight={
            percent != null ? (
              <span className="text-muted">
                step {sampling.progress_step} / {sampling.progress_total} ({percent}%)
              </span>
            ) : undefined
          }
        />
      )}

      {sampling.lora_paths.length > 0 && (
        <div className="space-y-1">
          <h2 className="text-sm font-medium text-muted">LoRA paths</h2>
          <ul className="text-sm text-text space-y-1 font-mono break-all">
            {sampling.lora_paths.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        </div>
      )}

      {sampling.status === "completed" && (
        <>
          <SweepGridViewer samplingId={id} status={sampling.status} />
          <LegacySamples samplingId={id} status={sampling.status} />
        </>
      )}

      {showLogs && (
        <LiveLogsPanel
          swrKey={`/samplings/${id}/logs`}
          fetcher={() => samplingsApi.getLogs(id, 500)}
          isRunning={isRunning}
          showLogs={showLogs}
          status={sampling.status}
          title="Sampling Logs"
        />
      )}

      {sampling.status === "failed" && sampling.error_message && (
        <div className="rounded-lg bg-error-muted border border-error/30 text-error px-4 py-3 text-sm">
          <strong>Error:</strong> {sampling.error_message}
        </div>
      )}
    </div>
  );
}
