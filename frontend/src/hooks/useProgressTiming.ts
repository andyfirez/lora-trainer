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
): ProgressTiming {
  const startRef = useRef<number | null>(null);
  const lastStepRef = useRef<number | null>(null);
  const lastStepTimeRef = useRef<number | null>(null);
  const [secondsPerIteration, setSecondsPerIteration] = useState<number | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    if (step == null || step < 0) {
      startRef.current = null;
      lastStepRef.current = null;
      lastStepTimeRef.current = null;
      setSecondsPerIteration(null);
      setElapsedSeconds(0);
      return;
    }

    const now = Date.now();
    if (startRef.current === null) {
      startRef.current = now;
      lastStepRef.current = step;
      lastStepTimeRef.current = now;
      return;
    }

    if (lastStepRef.current !== null && step > lastStepRef.current) {
      const deltaSteps = step - lastStepRef.current;
      const deltaTime = (now - (lastStepTimeRef.current ?? now)) / 1000;
      setSecondsPerIteration(deltaTime / deltaSteps);
      lastStepRef.current = step;
      lastStepTimeRef.current = now;
    }
  }, [step]);

  useEffect(() => {
    if (!active) {
      startRef.current = null;
      lastStepRef.current = null;
      lastStepTimeRef.current = null;
      setSecondsPerIteration(null);
      setElapsedSeconds(0);
    }
  }, [active]);

  useEffect(() => {
    if (!active || startRef.current === null) {
      return;
    }
    const tick = () => {
      if (startRef.current !== null) {
        setElapsedSeconds((Date.now() - startRef.current) / 1000);
      }
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [active, step]);

  return computeProgressTiming(step, total, elapsedSeconds, secondsPerIteration);
}
