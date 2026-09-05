"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import { samplingsApi } from "@/lib/api/samplings";
import { clearPlaygroundLoraState } from "@/lib/experimentUtils";
import { loadPlaygroundState, savePlaygroundState } from "@/lib/playgroundState";
import {
  resolveGenerateBusy,
  resolveGenerateLabel,
  type GeneratePrimaryLabel,
} from "@/lib/samplingGenerateLabel";
import { activePlaygroundJobs, mergeSessionIds } from "@/lib/samplingSession";
export type { GeneratePrimaryLabel };

function sameIds(left: number[], right: number[]): boolean {
  return left.length === right.length && left.every((id, index) => id === right[index]);
}

export function useSamplingJobs() {
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [optimistic, setOptimistic] = useState(false);
  const [sessionIds, setSessionIds] = useState<number[]>([]);
  const [sessionReady, setSessionReady] = useState(false);

  const { data: samplings, mutate: mutateSamplings } = useSWR(
    "/samplings",
    () => samplingsApi.list(),
    { refreshInterval: 1000 },
  );
  const active = activePlaygroundJobs(samplings);
  const myRunning = active.find((item) => item.status === "running");
  const myQueued = active.find((item) => item.status === "queued");

  const primaryLabel = resolveGenerateLabel({
    optimistic,
    sessionQueued: Boolean(myQueued),
    sessionRunning: Boolean(myRunning),
  });
  const cancellable = myRunning ?? myQueued;
  const primaryBusy = resolveGenerateBusy({
    label: primaryLabel,
    submitting,
  });

  const beginImmediate = () => {
    setOptimistic(true);
  };

  useEffect(() => {
    setSessionIds(loadPlaygroundState().sessionIds);
    setSessionReady(true);
  }, []);

  useEffect(() => {
    if (!samplings || !sessionReady) return;
    setSessionIds((ids) => {
      const next = mergeSessionIds(ids, samplings);
      return sameIds(ids, next) ? ids : next;
    });
  }, [samplings, sessionReady]);

  useEffect(() => {
    if (!sessionReady) return;
    const stored = loadPlaygroundState();
    if (sameIds(stored.sessionIds, sessionIds)) return;
    savePlaygroundState({ ...stored, sessionIds });
  }, [sessionIds, sessionReady]);

  useEffect(() => {
    if (!optimistic) return;
    if (myQueued || myRunning) setOptimistic(false);
  }, [optimistic, myQueued, myRunning]);

  const interrupt = async () => {
    if (!cancellable) return;
    await samplingsApi.cancel(cancellable.id);
    setOptimistic(false);
    await mutateSamplings();
  };

  const generate = async (config: Record<string, unknown>, options: { composeGrids: boolean }) => {
    setOptimistic(true);
    setSubmitting(true);
    setError(null);
    try {
      const withFlags = clearPlaygroundLoraState({ ...config, compose_grids: options.composeGrids });
      const created = await samplingsApi.generate(withFlags, [], {
        kind: "playground",
        composeGrids: options.composeGrids,
      });
      setSessionIds((ids) => mergeSessionIds([...ids, created.id], samplings));
      await mutateSamplings();
      return created;
    } catch (err) {
      setOptimistic(false);
      setError(err instanceof Error ? err.message : "Failed to start sampling");
      return null;
    } finally {
      setSubmitting(false);
    }
  };

  return {
    samplings,
    sessionIds,
    sessionReady,
    myRunning,
    myQueued,
    showProgress: optimistic || Boolean(myQueued) || Boolean(myRunning),
    primaryLabel,
    primaryBusy,
    optimistic,
    submitting,
    error,
    setError,
    beginImmediate,
    interrupt,
    generate,
  };
}
