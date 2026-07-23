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

export function computeMovingAverageWindow(total: number | null): number {
  if (total == null || total <= 0) {
    return Infinity;
  }
  return Math.max(1, Math.ceil(total * 0.15));
}

export function computeAverageStepDuration(
  durations: number[],
  windowSize: number,
): number | null {
  if (durations.length === 0) {
    return null;
  }
  const window = durations.slice(-windowSize);
  return window.reduce((sum, value) => sum + value, 0) / window.length;
}

export function computeProgressTiming(
  step: number | null,
  total: number | null,
  elapsedSeconds: number,
  secondsPerIteration: number | null,
  recentStepDurations: number[] = [],
): ProgressTiming {
  const completedSteps = step != null && step > 0 ? step : 0;
  const windowSize = computeMovingAverageWindow(total);
  const avgSecondsPerIteration =
    completedSteps >= 1
      ? computeAverageStepDuration(recentStepDurations, windowSize)
      : null;

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
