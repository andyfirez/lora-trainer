import type { LossPoint, LossResponse } from "@/types";

export type LossSeriesMap = Record<string, LossPoint[]>;

export function mergeLossSeries({
  previous,
  results,
  wantedKeys,
  isInitialLoad,
  lastStepByKey,
}: {
  previous: LossSeriesMap;
  results: LossResponse[];
  wantedKeys: string[];
  isInitialLoad: boolean;
  lastStepByKey: Record<string, number | null>;
}): { next: LossSeriesMap; lastStepByKey: Record<string, number | null> } {
  const next: LossSeriesMap = { ...previous };
  const updatedLastStep = { ...lastStepByKey };

  for (const result of results) {
    const key = result.key;
    const newPoints = (result.points ?? []).filter((point) => point.value !== null);

    if (isInitialLoad) {
      next[key] = newPoints;
    } else if (newPoints.length) {
      const existing = next[key] ?? [];
      const previousLast = existing.length ? existing[existing.length - 1].step : null;
      const filtered =
        previousLast == null ? newPoints : newPoints.filter((point) => point.step > previousLast);
      next[key] = filtered.length ? [...existing, ...filtered] : existing;
    } else {
      next[key] = next[key] ?? [];
    }

    const finalSeries = next[key] ?? [];
    updatedLastStep[key] = finalSeries.length
      ? finalSeries[finalSeries.length - 1].step
      : (updatedLastStep[key] ?? null);
  }

  for (const existingKey of Object.keys(next)) {
    if (!wantedKeys.includes(existingKey)) {
      delete next[existingKey];
      delete updatedLastStep[existingKey];
    }
  }

  return { next, lastStepByKey: updatedLastStep };
}
