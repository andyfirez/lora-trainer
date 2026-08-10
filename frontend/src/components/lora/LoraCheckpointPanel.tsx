"use client";

import Card, { CardTitle } from "@/components/ui/Card";
import type { LoraResponse } from "@/types";

interface LoraCheckpointPanelProps {
  lora: LoraResponse;
}

export default function LoraCheckpointPanel({ lora }: LoraCheckpointPanelProps) {
  const hasLast = Boolean(lora.last_checkpoint_path);
  const hasResume =
    lora.resume_checkpoint_path != null ||
    lora.resume_from_epoch != null ||
    lora.resume_from_step != null;

  if (!hasLast && !hasResume && !lora.can_resume) {
    return null;
  }

  return (
    <Card className="space-y-4">
      <CardTitle className="text-base">Checkpoints</CardTitle>

      {lora.can_resume && (
        <p className="text-xs text-muted">This run can be resumed from the latest checkpoint on disk.</p>
      )}

      {hasLast && (
        <div>
          <div className="text-xs text-muted mb-1">Last checkpoint</div>
          <code className="text-success text-xs break-all">{lora.last_checkpoint_path}</code>
          {(lora.last_checkpoint_epoch != null || lora.last_checkpoint_step != null) && (
            <div className="text-xs text-muted mt-1">
              {lora.last_checkpoint_epoch != null && <>epoch {lora.last_checkpoint_epoch}</>}
              {lora.last_checkpoint_step != null && <> · step {lora.last_checkpoint_step}</>}
            </div>
          )}
        </div>
      )}

      {hasResume && (
        <div className="pt-2 border-t border-border space-y-2">
          <div className="text-xs text-muted">Resume state</div>
          {lora.resume_checkpoint_path && (
            <code className="text-xs break-all block">{lora.resume_checkpoint_path}</code>
          )}
          {(lora.resume_from_epoch != null || lora.resume_from_step != null) && (
            <div className="text-xs text-muted">
              {lora.resume_from_epoch != null && <>from epoch {lora.resume_from_epoch}</>}
              {lora.resume_from_step != null && <> · step {lora.resume_from_step}</>}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
