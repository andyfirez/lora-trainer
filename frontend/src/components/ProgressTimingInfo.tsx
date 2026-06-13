"use client";

import { useProgressTiming } from "@/hooks/useProgressTiming";
import { formatDuration, formatIterationSeconds } from "@/lib/progressTiming";

interface Props {
  step: number | null;
  total: number | null;
  active: boolean;
  compact?: boolean;
  className?: string;
}

export default function ProgressTimingInfo({
  step,
  total,
  active,
  compact = false,
  className = "",
}: Props) {
  const timing = useProgressTiming(step, total, active);

  if (compact) {
    return (
      <div className={`text-[var(--muted)] text-xs ${className}`}>
        {formatIterationSeconds(timing.secondsPerIteration)}/step · {formatDuration(timing.elapsedSeconds)} elapsed
        {timing.etaSeconds != null && <> · ~{formatDuration(timing.etaSeconds)} left</>}
      </div>
    );
  }

  return (
    <div className={`flex flex-wrap gap-x-4 gap-y-1 text-xs text-[var(--muted)] ${className}`}>
      <span>
        per step: <span className="text-white">{formatIterationSeconds(timing.secondsPerIteration)}</span>
      </span>
      <span>
        elapsed: <span className="text-white">{formatDuration(timing.elapsedSeconds)}</span>
      </span>
      <span>
        ETA: <span className="text-white">{formatDuration(timing.etaSeconds)}</span>
      </span>
    </div>
  );
}
