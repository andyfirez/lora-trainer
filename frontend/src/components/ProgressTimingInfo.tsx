"use client";

import { useProgressTiming } from "@/hooks/useProgressTiming";
import { formatDuration, formatIterationSeconds } from "@/lib/progressTiming";

interface Props {
  step: number | null;
  total: number | null;
  active: boolean;
  elapsedSeconds?: number | null;
  compact?: boolean;
  className?: string;
}

export default function ProgressTimingInfo({
  step,
  total,
  active,
  elapsedSeconds = null,
  compact = false,
  className = "",
}: Props) {
  const timing = useProgressTiming(step, total, active, elapsedSeconds);

  if (compact) {
    return (
      <div className={`text-muted text-xs ${className}`}>
        {formatIterationSeconds(timing.avgSecondsPerIteration)}/step avg ·{" "}
        {formatIterationSeconds(timing.secondsPerIteration)}/step last ·{" "}
        {formatDuration(timing.elapsedSeconds)} elapsed
        {timing.etaSeconds != null && <> · ~{formatDuration(timing.etaSeconds)} left</>}
      </div>
    );
  }

  return (
    <div className={`flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted ${className}`}>
      <span>
        per step (avg):{" "}
        <span className="text-text">{formatIterationSeconds(timing.avgSecondsPerIteration)}</span>
      </span>
      <span>
        per step (last):{" "}
        <span className="text-text">{formatIterationSeconds(timing.secondsPerIteration)}</span>
      </span>
      <span>
        elapsed: <span className="text-text">{formatDuration(timing.elapsedSeconds)}</span>
      </span>
      <span>
        ETA: <span className="text-text">{formatDuration(timing.etaSeconds)}</span>
      </span>
    </div>
  );
}
