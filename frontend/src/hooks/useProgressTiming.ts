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
  backendElapsedSeconds: number | null,
): ProgressTiming {
  const lastStepRef = useRef<number | null>(null);
  const lastStepTimeRef = useRef<number | null>(null);
  const [secondsPerIteration, setSecondsPerIteration] = useState<number | null>(null);

  useEffect(() => {
    if (step == null || step < 0) {
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
  }, [step]);

  useEffect(() => {
    if (!active) {
      lastStepRef.current = null;
      lastStepTimeRef.current = null;
      setSecondsPerIteration(null);
    }
  }, [active]);

  const elapsedSeconds = backendElapsedSeconds ?? 0;

  return computeProgressTiming(step, total, elapsedSeconds, secondsPerIteration);
}
