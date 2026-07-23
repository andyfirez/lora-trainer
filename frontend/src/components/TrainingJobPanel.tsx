"use client";

import JobProgressBar from "@/components/JobProgressBar";
import JobLossGraph from "@/components/JobLossGraph";
import LiveLogsPanel from "@/components/LiveLogsPanel";
import JobsList from "@/components/JobsList";
import { jobsApi } from "@/lib/api/jobs";
import type { Job } from "@/types";

interface TrainingJobPanelProps {
  job: Job;
  lossGraphRunKey: number;
}

function progressPercent(step: number | null, total: number | null): number | null {
  if (step == null || total == null || total <= 0) {
    return null;
  }
  return Math.round((step / total) * 100);
}

export default function TrainingJobPanel({ job, lossGraphRunKey }: TrainingJobPanelProps) {
  const id = job.id;
  const training = job.training;
  const isRunning = job.status === "running";
  const showLogs = isRunning || job.status === "completed" || job.status === "failed" || job.status === "cancelled";
  const showLossGraph = showLogs;

  const trainStep = job.progress_step ?? 0;
  const trainPct =
    job.progress_total != null && job.progress_total > 0
      ? progressPercent(trainStep, job.progress_total)
      : null;

  return (
    <div className="space-y-6">
      {isRunning && training?.save_checkpoint_requested && (
        <div className="rounded-lg bg-warning/10 border border-warning/30 text-warning px-4 py-3 text-sm">
          Saving checkpoint before stopping…
        </div>
      )}

      {isRunning && trainPct != null && (
        <JobProgressBar
          title="Training Progress"
          step={trainStep}
          total={job.progress_total}
          percent={trainPct}
          active={isRunning}
          elapsedSeconds={job.elapsed_seconds}
          headerRight={
            <span className="text-muted">
              {training?.progress_epoch != null && training.progress_epoch > 0 && training.progress_epoch_total != null && (
                <>epoch {training.progress_epoch}/{training.progress_epoch_total} · </>
              )}
              step {trainStep} / {job.progress_total} ({trainPct}%)
            </span>
          }
          footer={
            (training?.progress_loss != null || training?.progress_avr_loss != null) ? (
              <div className="flex gap-4 text-xs text-muted">
                {training.progress_loss != null && (
                  <span>loss: <span className="text-text">{training.progress_loss.toFixed(4)}</span></span>
                )}
                {training.progress_avr_loss != null && (
                  <span>avr_loss: <span className="text-text">{training.progress_avr_loss.toFixed(4)}</span></span>
                )}
              </div>
            ) : undefined
          }
        />
      )}

      {showLossGraph && (
        <JobLossGraph jobId={id} isActive={isRunning} resetKey={String(lossGraphRunKey)} />
      )}

      <div className="space-y-4">
        <div>
          <h2 className="text-sm font-medium text-text">Sampling Jobs</h2>
          <p className="text-xs text-muted mt-1">
            Sampling runs created from this training job.
          </p>
        </div>
        <JobsList sourceJobId={id} />
      </div>

      {showLogs && (
        <LiveLogsPanel
          swrKey={`/jobs/${id}/logs`}
          fetcher={() => jobsApi.getLogs(id, 500)}
          isRunning={isRunning}
          showLogs={showLogs}
          status={job.status}
          title="Training Logs"
        />
      )}

      {job.status === "failed" && job.error_message && (
        <div className="rounded-lg bg-error-muted border border-error/30 text-error px-4 py-3 text-sm">
          <strong>Error:</strong> {job.error_message}
        </div>
      )}

      {job.output_path && (
        <div className="bg-surface rounded-xl border border-border p-4">
          <div className="text-xs text-muted mb-1">Output</div>
          <code className="text-success text-sm">{job.output_path}</code>
        </div>
      )}

      {training?.last_checkpoint_path && (
        <div className="bg-surface rounded-xl border border-border p-4">
          <div className="text-xs text-muted mb-1">Last Checkpoint</div>
          <code className="text-success text-sm break-all">{training.last_checkpoint_path}</code>
          {training.last_checkpoint_epoch != null && (
            <div className="text-xs text-muted mt-1">
              epoch {training.last_checkpoint_epoch}
              {training.last_checkpoint_step != null && ` · step ${training.last_checkpoint_step}`}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
