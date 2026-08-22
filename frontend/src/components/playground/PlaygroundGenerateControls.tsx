"use client";

import { Loader2 } from "lucide-react";
import Button from "@/components/ui/Button";
import type { GeneratePrimaryLabel } from "@/hooks/useSamplingJobs";

interface PlaygroundGenerateControlsProps {
  primaryLabel: GeneratePrimaryLabel;
  primaryBusy: boolean;
  onPrimary: () => void;
  error: string | null;
  className?: string;
}

export default function PlaygroundGenerateControls({
  primaryLabel,
  primaryBusy,
  onPrimary,
  error,
  className = "",
}: PlaygroundGenerateControlsProps) {
  return (
    <div className={`flex w-full shrink-0 flex-col ${className}`}>
      <div className="mb-1 h-4" aria-hidden />
      <Button
        variant={primaryLabel === "Interrupt" ? "danger" : "sampling"}
        className="h-10 w-full text-base"
        onClick={onPrimary}
        disabled={primaryBusy && primaryLabel !== "Interrupt"}
      >
        {primaryBusy && primaryLabel !== "Interrupt" ? (
          <Loader2 size={16} className="animate-spin" />
        ) : null}
        {primaryLabel}
      </Button>
      {error ? <p className="mt-2 text-xs text-error">{error}</p> : null}
    </div>
  );
}
