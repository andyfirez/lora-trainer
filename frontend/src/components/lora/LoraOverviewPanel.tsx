"use client";

import Card, { CardTitle } from "@/components/ui/Card";
import type { LoraResponse } from "@/types";

function formatElapsed(seconds: number | null | undefined): string | null {
  if (seconds == null || !Number.isFinite(seconds)) return null;
  const total = Math.floor(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

interface LoraOverviewPanelProps {
  lora: LoraResponse;
}

export default function LoraOverviewPanel({ lora }: LoraOverviewPanelProps) {
  const elapsed = formatElapsed(lora.elapsed_seconds ?? lora.accumulated_elapsed_seconds);

  return (
    <Card className="space-y-4">
      <CardTitle className="text-base">Overview</CardTitle>

      <dl className="space-y-3 text-sm">
        <div>
          <dt className="text-xs text-muted mb-0.5">Base model</dt>
          <dd className="text-text break-all">{lora.base_model_name}</dd>
        </div>
        <div>
          <dt className="text-xs text-muted mb-0.5">Created</dt>
          <dd className="text-text">{new Date(lora.created_at).toLocaleString()}</dd>
        </div>
        {elapsed && (
          <div>
            <dt className="text-xs text-muted mb-0.5">Elapsed</dt>
            <dd className="text-text">{elapsed}</dd>
          </div>
        )}
        {lora.status === "queued" && lora.queue_position != null && (
          <div>
            <dt className="text-xs text-muted mb-0.5">Queue position</dt>
            <dd className="text-text">#{lora.queue_position}</dd>
          </div>
        )}
        {lora.relative_path && (
          <div>
            <dt className="text-xs text-muted mb-0.5">Relative path</dt>
            <dd className="text-text break-all font-mono text-xs">{lora.relative_path}</dd>
          </div>
        )}
        {lora.resolved_weights_path && (
          <div>
            <dt className="text-xs text-muted mb-0.5">Weights</dt>
            <dd className="text-text break-all font-mono text-xs">{lora.resolved_weights_path}</dd>
          </div>
        )}
        {lora.resolved_work_dir && (
          <div>
            <dt className="text-xs text-muted mb-0.5">Work dir</dt>
            <dd className="text-text break-all font-mono text-xs">{lora.resolved_work_dir}</dd>
          </div>
        )}
        {lora.output_path && (
          <div>
            <dt className="text-xs text-muted mb-0.5">Output</dt>
            <dd className="text-text break-all font-mono text-xs">{lora.output_path}</dd>
          </div>
        )}
        {lora.log_path && (
          <div>
            <dt className="text-xs text-muted mb-0.5">Log file</dt>
            <dd className="text-text break-all font-mono text-xs">{lora.log_path}</dd>
          </div>
        )}
      </dl>
    </Card>
  );
}
