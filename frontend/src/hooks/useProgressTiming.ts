"use client";

import { useEffect, useRef, useState } from "react";
import {
  computeMovingAverageWindow,
  computeProgressTiming,
  type ProgressTiming,
} from "@/lib/progressTiming";

function trimStepDurations(durations: number[], total: number | null): number[] {
  const windowSize = computeMovingAverageWindow(total);
  if (!Number.isFinite(windowSize) || durations.length <= windowSize) {
    return durations;
  }
  return durations.slice(-windowSize);
}

export function useProgressTiming(
  step: number | null,
  total: number | null,
  active: boolean,
  elapsedSecondsFromBackend: number | null,
): ProgressTiming {
  const lastStepRef = useRef<number | null>(null);
  const lastStepTimeRef = useRef<number | null>(null);
  const stepDurationsRef = useRef<number[]>([]);
  const [secondsPerIteration, setSecondsPerIteration] = useState<number | null>(null);

  useEffect(() => {
    if (!active || step == null || step < 0) {
      lastStepRef.current = null;
      lastStepTimeRef.current = null;
      stepDurationsRef.current = [];
      setSecondsPerIteration(null);
      return;
    }

    const now = Date.now();
    if (lastStepRef.current === null) {
      lastStepRef.current = step;
      lastStepTimeRef.current = now;
      return;
    }

    if (step < lastStepRef.current) {
      lastStepRef.current = step;
      lastStepTimeRef.current = now;
      stepDurationsRef.current = [];
      setSecondsPerIteration(null);
      return;
    }

    if (step > lastStepRef.current) {
      const deltaSteps = step - lastStepRef.current;
      const deltaTime = (now - (lastStepTimeRef.current ?? now)) / 1000;
      const perStep = deltaTime / deltaSteps;
      const next = trimStepDurations(
        [...stepDurationsRef.current, ...Array(deltaSteps).fill(perStep)],
        total,
      );
      stepDurationsRef.current = next;
      setSecondsPerIteration(perStep);
      lastStepRef.current = step;
      lastStepTimeRef.current = now;
    }
  }, [active, step, total]);

  const elapsedSeconds = elapsedSecondsFromBackend ?? 0;

  return computeProgressTiming(
    step,
    total,
    elapsedSeconds,
    secondsPerIteration,
    stepDurationsRef.current,
  );
}
