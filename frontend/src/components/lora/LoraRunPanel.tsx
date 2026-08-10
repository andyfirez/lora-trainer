"use client";

import JobProgressBar from "@/components/JobProgressBar";
import LossGraph from "@/components/LossGraph";
import LiveLogsPanel from "@/components/LiveLogsPanel";
import { lorasApi } from "@/lib/api/loras";
import type { LoraResponse } from "@/types";

interface LoraRunPanelProps {
  lora: LoraResponse;
  lossGraphRunKey: number;
}

function progressPercent(step: number | null, total: number | null): number | null {
  if (step == null || total == null || total <= 0) {
    return null;
  }
  return Math.round((step / total) * 100);
}

export default function LoraRunPanel({ lora, lossGraphRunKey }: LoraRunPanelProps) {
  const id = lora.id;
  const isRunning = lora.status === "running";
  const showLogs =
    isRunning || lora.status === "completed" || lora.status === "failed" || lora.status === "cancelled";
  const showLossGraph = showLogs;

  const trainStep = lora.progress_step ?? 0;
  const trainPct =
    lora.progress_total != null && lora.progress_total > 0 ? progressPercent(trainStep, lora.progress_total) : null;

  return (
    <div className="space-y-6">
      {isRunning && lora.save_checkpoint_requested && (
        <div className="rounded-lg bg-warning/10 border border-warning/30 text-warning px-4 py-3 text-sm">
          Saving checkpoint before stopping…
        </div>
      )}

      {isRunning && trainPct != null && (
        <JobProgressBar
          title="Training Progress"
          step={trainStep}
          total={lora.progress_total}
          percent={trainPct}
          active={isRunning}
          elapsedSeconds={lora.elapsed_seconds}
          headerRight={
            <span className="text-muted">
              {lora.progress_epoch != null && lora.progress_epoch > 0 && lora.progress_epoch_total != null && (
                <>epoch {lora.progress_epoch}/{lora.progress_epoch_total} · </>
              )}
              step {trainStep} / {lora.progress_total} ({trainPct}%)
            </span>
          }
          footer={
            lora.progress_loss != null || lora.progress_avr_loss != null ? (
              <div className="flex gap-4 text-xs text-muted">
                {lora.progress_loss != null && (
                  <span>loss: <span className="text-text">{lora.progress_loss.toFixed(4)}</span></span>
                )}
                {lora.progress_avr_loss != null && (
                  <span>avr_loss: <span className="text-text">{lora.progress_avr_loss.toFixed(4)}</span></span>
                )}
              </div>
            ) : undefined
          }
        />
      )}

      {showLossGraph && <LossGraph loraId={id} isActive={isRunning} resetKey={String(lossGraphRunKey)} />}

      {showLogs && (
        <LiveLogsPanel
          swrKey={`/loras/${id}/logs`}
          fetcher={() => lorasApi.getLogs(id, 500)}
          isRunning={isRunning}
          showLogs={showLogs}
          status={lora.status}
          title="Training Logs"
        />
      )}

      {lora.status === "failed" && lora.error_message && (
        <div className="rounded-lg bg-error-muted border border-error/30 text-error px-4 py-3 text-sm">
          <strong>Error:</strong> {lora.error_message}
        </div>
      )}

      {lora.output_path && (
        <div className="bg-surface rounded-xl border border-border p-4">
          <div className="text-xs text-muted mb-1">Output</div>
          <code className="text-success text-sm">{lora.output_path}</code>
        </div>
      )}

      {lora.last_checkpoint_path && (
        <div className="bg-surface rounded-xl border border-border p-4">
          <div className="text-xs text-muted mb-1">Last Checkpoint</div>
          <code className="text-success text-sm break-all">{lora.last_checkpoint_path}</code>
          {lora.last_checkpoint_epoch != null && (
            <div className="text-xs text-muted mt-1">
              epoch {lora.last_checkpoint_epoch}
              {lora.last_checkpoint_step != null && ` · step ${lora.last_checkpoint_step}`}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
