export interface ProgressTiming {
  secondsPerIteration: number | null;
  avgSecondsPerIteration: number | null;
  elapsedSeconds: number;
  etaSeconds: number | null;
}

export function formatDuration(seconds: number | null): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) {
    return "—";
  }
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }
  if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return secs > 0 ? `${minutes}m ${secs}s` : `${minutes}m`;
  }
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.round((seconds % 3600) / 60);
  return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
}

export function formatIterationSeconds(seconds: number | null): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) {
    return "—";
  }
  if (seconds < 10) {
    return `${seconds.toFixed(1)}s`;
  }
  return `${Math.round(seconds)}s`;
}

export function computeProgressTiming(
  step: number | null,
  total: number | null,
  elapsedSeconds: number,
  secondsPerIteration: number | null,
): ProgressTiming {
  const completedSteps = step != null && step > 0 ? step : 0;
  const avgSecondsPerIteration =
    completedSteps > 0 && elapsedSeconds > 0 ? elapsedSeconds / completedSteps : null;

  const remaining =
    step != null && total != null && total > step ? total - step : null;
  const etaSeconds =
    remaining != null && avgSecondsPerIteration != null
      ? remaining * avgSecondsPerIteration
      : null;

  return {
    secondsPerIteration: step != null && step >= 2 ? secondsPerIteration : null,
    avgSecondsPerIteration: completedSteps >= 1 ? avgSecondsPerIteration : null,
    elapsedSeconds,
    etaSeconds: completedSteps >= 1 ? etaSeconds : null,
  };
}
