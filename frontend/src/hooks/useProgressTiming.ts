"use client";

import { useEffect, useRef, useState } from "react";
import {
  computeProgressTiming,
  type ProgressTiming,
} from "@/lib/progressTiming";

export function useProgressTiming(
  step: number | null,
  total: number | null,
  active: boolean,
  elapsedSecondsFromBackend: number | null,
): ProgressTiming {
  const lastStepRef = useRef<number | null>(null);
  const lastStepTimeRef = useRef<number | null>(null);
  const [secondsPerIteration, setSecondsPerIteration] = useState<number | null>(null);

  useEffect(() => {
    if (!active || step == null || step < 0) {
      lastStepRef.current = null;
      lastStepTimeRef.current = null;
      setSecondsPerIteration(null);
      return;
    }

    const now = Date.now();
    if (lastStepRef.current === null) {
      lastStepRef.current = step;
      lastStepTimeRef.current = now;
      return;
    }

    if (step > lastStepRef.current) {
      const deltaSteps = step - lastStepRef.current;
      const deltaTime = (now - (lastStepTimeRef.current ?? now)) / 1000;
      setSecondsPerIteration(deltaTime / deltaSteps);
      lastStepRef.current = step;
      lastStepTimeRef.current = now;
    }
  }, [active, step]);

  const elapsedSeconds = elapsedSecondsFromBackend ?? 0;

  return computeProgressTiming(step, total, elapsedSeconds, secondsPerIteration);
}
