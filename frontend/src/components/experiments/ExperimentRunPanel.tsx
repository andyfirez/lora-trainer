"use client";

import useSWR from "swr";
import JobProgressBar from "@/components/JobProgressBar";
import LiveLogsPanel from "@/components/LiveLogsPanel";
import ExperimentGridViewer from "@/components/experiments/ExperimentGridViewer";
import { samplingsApi } from "@/lib/api/samplings";
import { mediaUrl } from "@/lib/media";
import type { SamplingResponse } from "@/types";

interface ExperimentRunPanelProps {
  experiment: SamplingResponse;
  hideLogs?: boolean;
}

function progressPercent(step: number | null, total: number | null): number | null {
  if (step == null || total == null || total <= 0) return null;
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
          <a key={sample.path} href={mediaUrl(sample.url)} target="_blank" rel="noreferrer">
            <img
              src={mediaUrl(sample.url)}
              alt={sample.filename}
              className="rounded-lg border border-border object-cover aspect-square"
            />
          </a>
        ))}
      </div>
    </div>
  );
}

export default function ExperimentRunPanel({ experiment, hideLogs = false }: ExperimentRunPanelProps) {
  const id = experiment.id;
  const isRunning = experiment.status === "running";
  const isQueued = experiment.status === "queued";
  const showLogs =
    isRunning ||
    isQueued ||
    experiment.status === "completed" ||
    experiment.status === "failed" ||
    experiment.status === "cancelled";
  const percent = progressPercent(experiment.progress_step, experiment.progress_total);

  return (
    <div className="space-y-6">
      {isQueued && (
        <JobProgressBar
          title="Waiting in queue"
          step={null}
          total={null}
          percent={0}
          active={false}
          elapsedSeconds={experiment.elapsed_seconds}
          showBar={false}
          headerRight={
            experiment.queue_position != null ? (
              <span className="text-muted">Position #{experiment.queue_position}</span>
            ) : undefined
          }
        />
      )}

      {(isRunning || experiment.progress_status) && (
        <JobProgressBar
          title={experiment.progress_status ?? "Experiment"}
          step={experiment.progress_step}
          total={experiment.progress_total}
          percent={percent ?? 0}
          active={isRunning}
          elapsedSeconds={experiment.elapsed_seconds}
          barClassName="bg-sampling"
          showSpinner={isRunning}
          showBar={percent != null}
          headerRight={
            percent != null ? (
              <span className="text-muted">
                step {experiment.progress_step} / {experiment.progress_total} ({percent}%)
              </span>
            ) : undefined
          }
        />
      )}

      {experiment.lora_paths.length > 0 && (
        <div className="space-y-1">
          <h2 className="text-sm font-medium text-muted">LoRA paths</h2>
          <ul className="text-sm text-text space-y-1 font-mono break-all">
            {experiment.lora_paths.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        </div>
      )}

      {experiment.status === "completed" && (
        <>
          <ExperimentGridViewer samplingId={id} status={experiment.status} />
          <LegacySamples samplingId={id} status={experiment.status} />
        </>
      )}

      {showLogs && !hideLogs && (
        <LiveLogsPanel
          swrKey={`/samplings/${id}/logs`}
          fetcher={() => samplingsApi.getLogs(id, 500)}
          isRunning={isRunning}
          showLogs={showLogs}
          status={experiment.status}
          title="Experiment Logs"
        />
      )}

      {experiment.status === "failed" && experiment.error_message && (
        <div className="rounded-lg bg-error-muted border border-error/30 text-error px-4 py-3 text-sm">
          <strong>Error:</strong> {experiment.error_message}
        </div>
      )}
    </div>
  );
}
